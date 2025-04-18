# pylint: disable=not-context-manager, disable=protected-access

from typing import Any, Callable, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sympy import Union
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate, SVI
from pyro.distributions import constraints
from pyro.distributions import Dirichlet, Categorical, MultivariateNormal, Normal, Bernoulli  # type: ignore
from pyro.optim import Adam  # pylint: disable=no-name-in-module # type: ignore
from pyro.ops.indexing import Vindex
import pyro.poutine as poutine
from pyro.util import ignore_jit_warnings


# A GARCH model (Generalized Autoregressive Conditional Heteroskedasticity) is a time-series model
# typically used to describe and forecast volatility (or variance) that evolves over time. In
# financial contexts, it is frequently employed to capture the “clustering” effect of volatility:
# large swings in prices tend to be followed by large swings (of either sign), and small swings tend
# to be followed by small swings. Below is the high-level idea of GARCH(1,1), the most common form:
# Volatility changes over time: Instead of volatility (standard deviation) being constant, it is
# allowed to vary each period. We refer to the variance of period t as sigma_t^2. Recurrence
# relation for volatility: A GARCH(1,1) model can be written as:

#           sigma_t^2 = omega + alpha *e_(t-1)^2 + beta * sigma_(t-1)^2

# • omega is a positive constant or “baseline” volatility term.
# • alpha measures how strongly volatility reacts to the magnitude of previous shocks e_(t-1). That
# shock is often r_(t-1), the return at time t-1, or sometimes (r_(t-1) - mu_(t-1)) if using a mean
# model.
# • beta measures how persistently volatility carries forward from one period to the next.
# • e_(t-1)^2 is the squared residual (or shock) from the previous time step.
# • sigma_(t-1)^2 is the variance from the previous time step.

# Stationarity requirement: For GARCH(1,1) to remain stable (i.e., not explode to infinite
# variance), we generally require alpha + beta < 1.

# Interpretation: Intuitively, if a large shock occurred in the previous step (e_(t-1)^2 is large),
# the model increases the volatility for the current step. Over time, if there are no large shocks,
# sigma_t^2 decays according to beta until it approaches its long-term mean level implied by omega.

# Capturing volatility clustering: Because GARCH ties the current volatility partly to the previous
# squared shock, if market returns were volatile recently, the model automatically forecasts higher
# volatility in subsequent periods, matching the clustered volatility often seen in real markets.

# By extending beyond GARCH(1,1) to include additional lags or other distributions for e_t (e.g.,
# t-distribution), you can capture more complex volatility patterns. However, GARCH(1,1) remains a
# popular baseline for modeling volatility in finance due to its simplicity and effectiveness at
# detecting volatility clustering.


def ar_garch_model_student_t(
    returns: Optional[torch.Tensor] = None,
    T: int = 3,
    prior_predictive_checks: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """
    AR(1)-GARCH(1,1) model with Student-T innovations, modified to run on CPU or GPU.

    Parameters
    ----------
    returns : torch.Tensor or None
        Shape [T], containing observed returns. If None, we sample returns from the prior/posterior predictive.
    T : int
        Number of time steps to model. If returns is provided, T is overridden by returns.shape[0].
    prior_predictive_checks : bool
        If True, observations are set to None (sampling from prior or posterior predictive).
    device : torch.device
        The device on which to run the computations (e.g., torch.device("cpu") or torch.device("cuda")).
    """

    # Move data (if any) to the chosen device
    if returns is not None:
        returns = returns.to(device)
        T = returns.shape[0]

    # Helper to ensure parameters are on the same device
    def param_tensor(x):
        return torch.tensor(x, device=device)

    # 1) GARCH parameters
    garch_omega = pyro.sample("garch_omega", dist.Exponential(param_tensor(10.0)))

    alpha_beta_sum = pyro.sample("alpha_beta_sum", dist.Beta(param_tensor(9.0), param_tensor(1.0)))
    alpha_frac = pyro.sample("alpha_frac", dist.Beta(param_tensor(2.0), param_tensor(2.0)))
    garch_alpha = alpha_beta_sum * alpha_frac
    garch_beta = alpha_beta_sum * (1.0 - alpha_frac)

    # 2) AR(1) parameter
    phi = pyro.sample("phi", dist.Normal(param_tensor(0.0), param_tensor(1.0)))

    # 3) Initial volatility
    sigma_init = pyro.sample("garch_sigma_init", dist.Exponential(param_tensor(10.0)))

    # 4) Degrees of freedom for Student-T (df > 2)
    df = pyro.sample("degrees_of_freedom", dist.Exponential(param_tensor(1.0))) + param_tensor(2.0)

    # Initialize recursion variables on the correct device
    sigma_prev = sigma_init
    e_prev = param_tensor(0.0)
    r_prev = param_tensor(0.0)

    with pyro.markov():
        for t in range(T):
            if t == 0:
                # No GARCH recursion has been applied yet
                sigma_t = sigma_init
                mean_t = param_tensor(0.0)
            else:
                # GARCH(1,1) update for sigma_t
                sigma_t = torch.sqrt(
                    garch_omega + garch_alpha * (e_prev**2) + garch_beta * (sigma_prev**2)
                )
                mean_t = phi * r_prev

            # Either observe (condition) or sample the return
            obs = None
            if not prior_predictive_checks and returns is not None:
                obs = returns[t]

            r_t = pyro.sample(f"r_{t}", dist.StudentT(df, mean_t, sigma_t), obs=obs)

            # Update for the next step
            e_prev = r_t - mean_t  # e_t
            sigma_prev = sigma_t
            r_prev = r_t


def ar_garch_model_student_t_multi_asset_partial_pooling(
    returns,  # shape [batch_size, max_T]
    lengths,  # shape [batch_size,] giving lengths per asset
    args,  # expects args.jit, and other control flags/settings
    prior_predictive_checks: bool = False,
    device: torch.device = torch.device("cpu"),
    batch_size=None,  # support for mini-batched learning
):
    """
    Parallel AR(1)-GARCH(1,1) model with Student-T innovations
    with partial pooling/hierarchical structure for parameters
    (each asset draws its own set of parameters from global hyperpriors).
    Designed for Pyro JIT & masking. Can handle assets with differing available lengths.

    Other comments as before...
    """

    # Move to correct device
    returns = returns.to(device)
    lengths = lengths.to(device)
    num_assets, max_T = returns.shape

    # ------------------- HIERARCHICAL PRIORS (hyperpriors for group/global parameters) --------------------

    # 1) GARCH omega
    omega_mu = pyro.sample("omega_mu", dist.Exponential(torch.tensor(10.0, device=device)))
    omega_sigma = pyro.sample("omega_sigma", dist.Exponential(torch.tensor(10.0, device=device)))

    # 2) alpha_beta_sum
    ab_sum_mu = pyro.sample(
        "ab_sum_mu", dist.Beta(torch.tensor(9.0, device=device), torch.tensor(1.0, device=device))
    )
    ab_sum_sigma = pyro.sample("ab_sum_sigma", dist.Exponential(torch.tensor(5.0, device=device)))

    # 3) alpha_frac
    ab_frac_mu = pyro.sample(
        "ab_frac_mu", dist.Beta(torch.tensor(2.0, device=device), torch.tensor(2.0, device=device))
    )
    ab_frac_sigma = pyro.sample("ab_frac_sigma", dist.Exponential(torch.tensor(5.0, device=device)))

    # 4) AR(1) phi
    phi_mu = pyro.sample(
        "phi_mu", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    )
    phi_sigma = pyro.sample("phi_sigma", dist.Exponential(torch.tensor(1.0, device=device)))

    # 5) Initial GARCH sigma
    sigma_init_mu = pyro.sample(
        "sigma_init_mu", dist.Exponential(torch.tensor(10.0, device=device))
    )
    sigma_init_sigma = pyro.sample(
        "sigma_init_sigma", dist.Exponential(torch.tensor(10.0, device=device))
    )

    # 6) Degrees of freedom for Student-T (constrained to >2)
    df_mu = pyro.sample("df_mu", dist.Exponential(torch.tensor(1.0, device=device)))
    df_sigma = pyro.sample("df_sigma", dist.Exponential(torch.tensor(1.0, device=device)))

    # ---------------------- PER-ASSET PARAMETERS (these are now random, per asset) -------------------------
    with ignore_jit_warnings():
        with pyro.plate("assets", num_assets, batch_size, dim=-2) as batch:
            batch_size_local = batch.shape[0] if batch is not None else num_assets

            asset_lengths = lengths[batch]  # [batch_size_local]
            asset_returns = returns[batch]  # [batch_size_local, max_T]

            # ASSET-SPECIFIC parameters from the group/hyperpriors (partial pooling!)
            garch_omega = pyro.sample(
                "garch_omega", dist.Normal(omega_mu, omega_sigma).expand([batch_size_local])
            )
            garch_omega = garch_omega.clamp(min=1e-4)

            alpha_beta_sum = pyro.sample(
                "alpha_beta_sum", dist.Normal(ab_sum_mu, ab_sum_sigma).expand([batch_size_local])
            )
            alpha_beta_sum = alpha_beta_sum.clamp(min=1e-4, max=1 - 1e-4)

            alpha_frac = pyro.sample(
                "alpha_frac", dist.Normal(ab_frac_mu, ab_frac_sigma).expand([batch_size_local])
            )
            alpha_frac = alpha_frac.clamp(min=1e-4, max=1 - 1e-4)

            garch_alpha = alpha_beta_sum * alpha_frac
            garch_beta = alpha_beta_sum * (1.0 - alpha_frac)

            phi = pyro.sample("phi", dist.Normal(phi_mu, phi_sigma).expand([batch_size_local]))
            # phi is unconstrained, often fine for AR(1) as long as you keep inference stable

            sigma_init = pyro.sample(
                "garch_sigma_init",
                dist.Normal(sigma_init_mu, sigma_init_sigma).expand([batch_size_local]),
            )
            sigma_init = sigma_init.clamp(min=1e-4)

            df = pyro.sample(
                "degrees_of_freedom", dist.Normal(df_mu, df_sigma).expand([batch_size_local])
            )
            df = df.clamp(min=2.05)

            # Vectorized GARCH/AR recursion
            sigma_prev = sigma_init  # [batch_size_local]
            e_prev = torch.zeros(batch_size_local, device=device)
            r_prev = torch.zeros(batch_size_local, device=device)

            for t in pyro.markov(
                range(max_T if getattr(args, "jit", False) else asset_lengths.max().item())
            ):
                valid_mask = t < asset_lengths  # [batch_size_local]

                if t == 0:
                    sigma_t = sigma_prev
                    mean_t = torch.zeros_like(sigma_prev)
                else:
                    sigma_t = torch.sqrt(
                        garch_omega + garch_alpha * (e_prev**2) + garch_beta * (sigma_prev**2)
                    )
                    mean_t = phi * r_prev

                obs = None
                if not prior_predictive_checks and asset_returns is not None:
                    obs = asset_returns[:, t]  # [batch_size_local]

                with poutine.mask(mask=valid_mask):
                    r_t = pyro.sample(f"r_{t}", dist.StudentT(df, mean_t, sigma_t), obs=obs)

                e_prev = r_t - mean_t
                sigma_prev = sigma_t
                r_prev = r_t


# Usage requires a wrapper for args (so you can pass args.jit), for example:
# class Args: pass
# args = Args(); args.jit = True
# returns: padded data [num_assets, max_T], lengths: [num_assets]


if __name__ == "__main__":
    pyro.render_model(
        ar_garch_model_student_t,
        model_args=(None),
        filename="ar_garch_model_StudT.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )

    # Dummy data with one asset, 3 time steps
    dummy_returns = torch.zeros(1, 3)
    dummy_lengths = torch.tensor([3])

    # Dummy args
    @dataclass
    class Args:
        jit: bool = False
        # You can add other model control flags or params as needed
        # e.g. batch_size: int = None

    args = Args()
    args = Args(jit=False)  # Set jit=True if you want static structure

    pyro.render_model(
        ar_garch_model_student_t_multi_asset_partial_pooling,
        model_args=(dummy_returns, dummy_lengths, args),
        filename="ar_garch_StudT_multiassets_partialpool.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    ).unflatten(stagger=1)
