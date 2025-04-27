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
from helpers import debug_shape


# Model Equations:
# Mean (AR(1)):
# mean_t = phi * r_{t-1}

# Conditional variance (GARCH(1,1)):
# sigma_t^2 = omega + alpha * (r_{t-1} - mean_{t-1})^2 + beta * sigma_{t-1}^2

# Reparameterization for stability:
# alpha = alpha_beta_sum * alpha_frac
# beta = alpha_beta_sum * (1 - alpha_frac)
# where 0 < alpha_beta_sum < 1 (stationarity enforced automatically)

# Likelihood (Student-T noise):
# r_t ~ StudentT(df, mean_t, sigma_t)
# Typical priors (for Bayesian version):
# omega ~ LogNormal(0, 1) # Must be >0
# alpha_beta_sum ~ Beta(2, 2) # (0,1), so alpha+beta < 1
# alpha_frac ~ Beta(2, 2) # (0,1)
# phi ~ Normal(0, 1)
# sigma0 ~ LogNormal(0, 1) # Initial variance
# df ~ LogNormal(1, 0.3) + 2 # Degrees of freedom, >2


def ar_garch1_studentt_model(*args, **kwargs):
    """
    Dynamic Bayesian Network for AR(1)-GARCH(1,1) with Student-T innovations.
    - Handles missing data (NaN) in returns.
    - Parameters are reparametrized for GARCH stability.
    """
    device = kwargs.get("device", "cpu")
    # print(f"\nMODEL DEBUG - device:{device}")
    returns = args[0]
    T = returns.shape[0]

    # ===== PRIORS ON MODEL PARAMETERS =====
    omega = pyro.sample(
        "omega",
        dist.LogNormal(
            torch.full((), 0.0, device=device),
            torch.full((), 1.0, device=device),
        ),
    )
    alpha_beta_sum = pyro.sample(
        "alpha_beta_sum",
        dist.Beta(
            torch.full((), 2.0, device=device),
            torch.full((), 2.0, device=device),
        ),
    )
    alpha_frac = pyro.sample(
        "alpha_frac",
        dist.Beta(
            torch.full((), 2.0, device=device),
            torch.full((), 2.0, device=device),
        ),
    )
    phi = pyro.sample(
        "phi",
        dist.Normal(
            torch.full((), 0.0, device=device),
            torch.full((), 1.0, device=device),
        ),
    )
    sigma0 = pyro.sample(
        "sigma0",
        dist.LogNormal(
            torch.full((), 0.0, device=device),
            torch.full((), 1.0, device=device),
        ),
    )
    nu = (
        pyro.sample(
            "nu_minus2",
            dist.LogNormal(
                torch.full((), 1.0, device=device),
                torch.full((), 0.3, device=device),
            ),
        )
        + 2.0
    )
    # ===== CONSTRUCT GARCH PARAMETERS (enforces stability) =====
    garch_alpha = alpha_beta_sum * alpha_frac
    garch_beta = alpha_beta_sum * (1.0 - alpha_frac)
    # ===== INITIALIZE TEMPORAL STATE =====
    sigma_prev = sigma0
    r_prev = torch.zeros((), device=device)
    e_prev = torch.zeros((), device=device)
    # ===== MAIN TIME LOOP =====
    for t in pyro.markov(range(T)):
        # ----- GARCH: Compute conditional std at t -----
        sigma_t = (
            (
                omega + garch_alpha * e_prev**2 + garch_beta * sigma_prev**2
            ).sqrt()
            if t > 0
            else sigma_prev
        )
        # ----- AR(1): Compute conditional mean at t -----
        mean_t = phi * r_prev if t > 0 else torch.zeros(())
        # ----- Masking for Missing Data -----
        curr_obs = None if torch.isnan(returns[t]) else returns[t]
        # ----- Observation Model -----
        r_t = pyro.sample(
            f"r_{t}",
            dist.StudentT(nu, mean_t, sigma_t),
            obs=curr_obs,
        )
        # ----- Bookkeeping: Update state for next time -----
        e_prev = r_t - mean_t  # Innovation (residual)
        r_prev = r_t  # Last return
        sigma_prev = sigma_t  # Last conditional std


if __name__ == "__main__":
    # Dummy data: one asset, three time steps
    dummy_returns = torch.zeros(3)

    # You'll need to install graphviz to get graphical output!
    pyro.render_model(
        ar_garch1_studentt_model,
        model_args=(dummy_returns,),
        filename="model_01.jpg",
        render_params=True,
        render_distributions=True,
    )
