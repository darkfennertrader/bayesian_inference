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

# 1. High-Level Explanation
# Bayesian hierarchical AR(1)-GARCH(1,1) with Student-T innovations You model a panel
# (multiple-assets) of time series using a vectorized AR(1)-GARCH(1,1) model with Student-T
# innovations and partial pooling (hierarchical priors). This allows each asset to have its own
# parameters (omega, alpha, beta, AR(1) phi, etc.) drawn from learned global priors (hyperpriors),
# but encourages sharing statistical strength across assets.

# Each asset’s returns are:
# AR(1): Has autocorrelation GARCH(1,1): Time-varying volatility Student-T innovations: Fat tails
# Partial pooling: Use hyperpriors to share info but allow for idiosyncratic behavior Flexible
# weighting: Each asset and time has a decay weight (good for non-stationarity)


# 2. Model Structure
# returns: matrix [assets, max_T] of returns (observed, some padding possible)
# lengths: length for each asset (to mask paddings) You draw a suite of global priors for each
# parameter family (hierarchical Bayesian) For each asset (possibly mini-batched), draw individual
# parameters from the global priors ("partial pooling") Each time step, propagate the AR(1)-GARCH
# state and condition on observations, using temporal and per-asset weights Each asset’s returns are
# observed with Student-T (allowing fatter tails than Gaussian)

# 3. Global Priors/Hyperpriors
# omega_mu/sigma: Hyperpriors for GARCH omega (constant term)
# ab_sum_a/b_hyper: Beta hyperpriors for alpha + beta sum
# ab_frac_a/b_hyper: Beta hyperpriors for fraction of alpha to alpha + beta
# phi_mu/sigma: Hyperpriors for AR(1) coefficient ...
# (same idea for sigma_init and degrees_of_freedom)
# lambda_decay: Controls time-weighting via exponential decay
# per time period

# 4. Partial pooling (“plate” per asset)
# A per-asset garch_omega from the global (Normal, clamped > 0)
# Proper Beta distributed alpha_beta_sum (in (0,1))
# alpha_frac (in (0,1)), both with hierarchical Beta priors
# Compose GARCH parameters: garch_alpha = alpha_beta_sum * alpha_frac, garch_beta = alpha_beta_sum *
# (1-alpha_frac)
# AR(1) phi per asset, and initial sigma
# Student-T df, clamped > 2 for validity
# Per-asset likelihood weight (Beta)

# 5. Recursive State-per-time-point with masks
# Burn-in first step, else use true recursions for AR-GARCH
# Compute per-t log-likelihood on the observations
# Per-asset weights × time-wise decay (exponential in distance from present)
### Use pyro.factor to add weighted log-likelihood for each time step
### Masking: Only update loss/state if t < lengths[asset]
# Sample the latent state (even when observed! This is required by Pyro)

# Model Parameters:  13 + 7 × n_assets


def ar_garch_model_student_t_multi_asset_partial_pooling(
    returns,
    lengths,
    args,  # expects args.jit, and other control flags/settings
    prior_predictive_checks: bool = False,
    device: torch.device = torch.device("cpu")
):
    """
    Parallel AR(1)-GARCH(1,1) model with Student-T innovations
    with partial pooling/hierarchical structure for parameters
    (each asset draws its own set of parameters from global hyperpriors).

    Key modification: For (0,1) parameters (such as alpha_beta_sum, alpha_frac),
    use proper Beta partial pooling, not clamped Normals.

    Designed for Pyro JIT & masking. Can handle assets with differing available lengths.
    """

    # Move to correct device
    returns = returns.to(device)
    lengths = lengths.to(device)
    batch_size, max_T = returns.shape

    # ------------------- HIERARCHICAL PRIORS (hyperpriors for group/global parameters) --------------------

    # 1) GARCH omega (positive, not [0,1])
    omega_mu = pyro.sample("omega_mu", dist.Exponential(torch.tensor(1.0, device=device)))
    omega_sigma = pyro.sample("omega_sigma", dist.Exponential(torch.tensor(1.0, device=device)))

    # 2) alpha_beta_sum ~ Beta(a, b) hierarchy
    ab_sum_a_hyper = pyro.sample(
        "ab_sum_a_hyper", dist.Exponential(torch.tensor(2.0, device=device))
    )
    ab_sum_b_hyper = pyro.sample(
        "ab_sum_b_hyper", dist.Exponential(torch.tensor(2.0, device=device))
    )
    # 3) alpha_frac ~ Beta(a, b) hierarchy
    ab_frac_a_hyper = pyro.sample(
        "ab_frac_a_hyper", dist.Exponential(torch.tensor(2.0, device=device))
    )
    ab_frac_b_hyper = pyro.sample(
        "ab_frac_b_hyper", dist.Exponential(torch.tensor(2.0, device=device))
    )

    # 4) AR(1) phi (unconstrained)
    phi_mu = pyro.sample(
        "phi_mu", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    )
    phi_sigma = pyro.sample("phi_sigma", dist.Exponential(torch.tensor(1.0, device=device)))

    # 5) Initial GARCH sigma (positive)
    sigma_init_mu = pyro.sample(
        "sigma_init_mu", dist.Exponential(torch.tensor(10.0, device=device))
    )
    sigma_init_sigma = pyro.sample(
        "sigma_init_sigma", dist.Exponential(torch.tensor(10.0, device=device))
    )

    # 6) Degrees of freedom for Student-T (constrained to >2)
    df_mu = pyro.sample("df_mu", dist.Exponential(torch.tensor(1.0, device=device)))
    df_sigma = pyro.sample("df_sigma", dist.Exponential(torch.tensor(1.0, device=device)))

    # Global decay parameter for time-weighting - Values in (0,1): how rapidly to forget the past
    lambda_decay = pyro.sample(
        "lambda_decay",
        dist.Beta(torch.tensor(2.0, device=device), torch.tensor(2.0, device=device)),
    )

    # ---------------------- PER-ASSET PARAMETERS  -------------------------
    with ignore_jit_warnings():
        with pyro.plate("assets", batch_size, batch_size, dim=-2) as batch:
            asset_lengths = lengths  # already pre-selected [batch_size]
            asset_returns = returns  # already pre-selected [batch_size, max_T]

            # ASSET-SPECIFIC parameters from the group/hyperpriors (partial pooling!)

            # GARCH omega; positive, partial pooling via Normal
            garch_omega = pyro.sample(
                "garch_omega", dist.Normal(omega_mu, omega_sigma).expand([batch_size])
            )
            garch_omega = garch_omega.clamp(min=1e-4)  # safety

            # --- MODIFIED: Proper Beta partial pooling for alpha_beta_sum in (0,1) ---
            alpha_beta_sum = pyro.sample(
                "alpha_beta_sum",
                dist.Beta(ab_sum_a_hyper, ab_sum_b_hyper).expand([batch_size]),
            )

            # --- MODIFIED: Proper Beta partial pooling for alpha_frac in (0,1) ---
            alpha_frac = pyro.sample(
                "alpha_frac", dist.Beta(ab_frac_a_hyper, ab_frac_b_hyper).expand([batch_size])
            )

            # reparameterize
            garch_alpha = alpha_beta_sum * alpha_frac
            garch_beta = alpha_beta_sum * (1.0 - alpha_frac)

            # AR(1) phi (no change)
            phi = pyro.sample("phi", dist.Normal(phi_mu, phi_sigma).expand([batch_size]))

            # Initial GARCH sigma (positive)
            sigma_init = pyro.sample(
                "garch_sigma_init",
                dist.Normal(sigma_init_mu, sigma_init_sigma).expand([batch_size]),
            )
            sigma_init = sigma_init.clamp(min=1e-4)  # safety

            # Student-T dof (must be >2)
            df = pyro.sample(
                "degrees_of_freedom", dist.Normal(df_mu, df_sigma).expand([batch_size])
            )
            df = df.clamp(min=2.05)

            # Per-asset likelihood weight
            asset_weight = pyro.sample(
                "asset_weight",
                dist.Beta(
                    torch.tensor(2.0, device=device), torch.tensor(2.0, device=device)
                ).expand([asset_returns.shape[0]]),
            )

            # Vectorized GARCH/AR recursion
            sigma_prev = sigma_init  # [batch_size_local]
            e_prev = torch.zeros(batch_size, device=device)
            r_prev = torch.zeros(batch_size, device=device)

            for t in pyro.markov(
                range(max_T if getattr(args, "jit", False) else asset_lengths.max().item())
            ):
                valid_mask = t < asset_lengths  # [batch_size_local]
                decay_exponent = max_T - t - 1

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

                # --- Core: apply both per-asset and temporal weighting ---
                # Each point's log likelihood is: per-asset weight × decayed by time index
                if obs is not None:
                    # Calculate log likelihood "by hand" for control
                    log_prob = dist.StudentT(df, mean_t, sigma_t).log_prob(
                        obs
                    )  # shape [batch_size_local]
                    # Both per-asset and temporal scaling. Make sure shapes match!
                    combined_weight = asset_weight * (lambda_decay**decay_exponent)
                    # Apply mask (valid times only)
                    weighted_log_prob = torch.where(
                        valid_mask, combined_weight * log_prob, torch.zeros_like(log_prob)
                    )
                    # Use pyro.factor to modify SVI loss accordingly
                    pyro.factor(f"weighted_decay_{t}", weighted_log_prob)
                    # Comment: This gives the model flexibility to forget irrelevant *history* and *assets*.

                # Bookkeeping and recursion
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
