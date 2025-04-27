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


# ==== Device and Initializations (put these near top of your script) ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For GARCH model stability, typical:
OMEGA_LOC_INIT = torch.full((), -4.0, device=DEVICE)  # log(0.01)
SIGMA0_LOC_INIT = torch.full((), -2.0, device=DEVICE)  # log(0.13)
NU_M2_LOC_INIT = torch.full((), 1.0, device=DEVICE)  # log(2.7)
OMEGA_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
AB_SUM_A_INIT = torch.full((), 2.0, device=DEVICE)
AB_SUM_B_INIT = torch.full((), 2.0, device=DEVICE)
AB_FRAC_A_INIT = torch.full((), 2.0, device=DEVICE)
AB_FRAC_B_INIT = torch.full((), 2.0, device=DEVICE)
PHI_LOC_INIT = torch.full((), 0.0, device=DEVICE)
PHI_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
SIGMA0_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
NU_M2_SCALE_INIT = torch.full((), 0.1, device=DEVICE)


def ar_garch2_studentt_guide(returns, **kwargs):
    """
    Hierarchical variational guide for multi-asset AR(1)-GARCH(1,1)-StudentT model.

    Implements:
    - **Global ("hyper") variational parameters:** point estimates for shared hyperpriors.
    - **Asset-specific latent parameter vectors:** modeled via a hierarchical
      (low-rank) multivariate Gaussian for structured covariance across assets.
    - **Partial pooling:** Information shared via a global mean plus asset-level offsets.
    - **Parameter transforms:** Ensures constraints/positivity for GARCH/Student-T.

    The structure and naming match the model for automatic SVI.

    Arguments:
        returns:   [T, N_assets] observed returns, required for shape info
        **kwargs:  Accepts 'device' (cpu/gpu, optional), 'r' (low-rank size for Mvn)
    """
    print("\nGUIDE:")

    # -------------- DEVICE, SHAPES, PARAMETER SIZES -----------------
    # Device: where to store Pyro params (CPU/GPU)
    device = kwargs.get("device", torch.device("cpu"))
    returns = returns.to(device)

    # T: time steps, N_assets: assets in universe (52 in your case)
    T, N_assets = returns.shape

    # The latent parameter vector per asset has six dimensions:
    #   omega, alpha_beta_sum, alpha_frac, phi, sigma0, nu_minus2
    param_dim = 6

    # r: low-rank factor dimension for the global covariance structure.
    #   Typical values: 2, 3; balances statistical expressivity & efficiency.
    r = kwargs.get("r", 3)

    # ==================================================================
    # 1. GLOBAL GUIDES (ALL Pyro param() here are optimized by SVI!)
    # ==================================================================
    # Initialize point-mass (Delta) guide parameters for the model's global "hyperpriors":
    # These control means and scales for the asset-specific parameters.

    omega_loc_q = pyro.param("omega_loc_q", OMEGA_LOC_INIT.clone())
    omega_scale_q = pyro.param(
        "omega_scale_q",
        OMEGA_SCALE_INIT.clone(),
        constraint=constraints.positive,
    )
    ab_sum_a_q = pyro.param(
        "ab_sum_a_q", AB_SUM_A_INIT.clone(), constraint=constraints.positive
    )
    ab_sum_b_q = pyro.param(
        "ab_sum_b_q", AB_SUM_B_INIT.clone(), constraint=constraints.positive
    )
    ab_frac_a_q = pyro.param(
        "ab_frac_a_q", AB_FRAC_A_INIT.clone(), constraint=constraints.positive
    )
    ab_frac_b_q = pyro.param(
        "ab_frac_b_q", AB_FRAC_B_INIT.clone(), constraint=constraints.positive
    )
    phi_loc_q = pyro.param("phi_loc_q", PHI_LOC_INIT.clone())
    phi_scale_q = pyro.param(
        "phi_scale_q", PHI_SCALE_INIT.clone(), constraint=constraints.positive
    )
    sigma0_loc_q = pyro.param("sigma0_loc_q", SIGMA0_LOC_INIT.clone())
    sigma0_scale_q = pyro.param(
        "sigma0_scale_q",
        SIGMA0_SCALE_INIT.clone(),
        constraint=constraints.positive,
    )
    nu_m2_loc_q = pyro.param("nu_m2_loc_q", NU_M2_LOC_INIT.clone())
    nu_m2_scale_q = pyro.param(
        "nu_m2_scale_q",
        NU_M2_SCALE_INIT.clone(),
        constraint=constraints.positive,
    )

    # For each global hyperprior in the model, provide a (Dirac) point-value in the guide.
    # SVI will optimize these "central tendency/scatter" parameters during training.
    pyro.sample("omega_loc", dist.Delta(omega_loc_q))
    pyro.sample("omega_scale", dist.Delta(omega_scale_q))
    pyro.sample("ab_sum_a", dist.Delta(ab_sum_a_q))
    pyro.sample("ab_sum_b", dist.Delta(ab_sum_b_q))
    pyro.sample("ab_frac_a", dist.Delta(ab_frac_a_q))
    pyro.sample("ab_frac_b", dist.Delta(ab_frac_b_q))
    pyro.sample("phi_loc", dist.Delta(phi_loc_q))
    pyro.sample("phi_scale", dist.Delta(phi_scale_q))
    pyro.sample("sigma0_loc", dist.Delta(sigma0_loc_q))
    pyro.sample("sigma0_scale", dist.Delta(sigma0_scale_q))
    pyro.sample("nu_m2_loc", dist.Delta(nu_m2_loc_q))
    pyro.sample("nu_m2_scale", dist.Delta(nu_m2_scale_q))

    # ==================================================================
    # 2. HIERARCHICAL MULTIVARIATE STRUCTURE FOR ASSET PARAMETERS
    # ==================================================================
    # For parameter vectors theta_i (per asset):
    #   p(theta_i) ~ MVN(global location, [low-rank shared factor + diag noise])
    #   Each theta_i = [log omega, logit ab_sum, logit ab_frac, phi, log sigma0, log nu-2]
    #
    # This enables learning shared dynamics across assets, but each can deviate from the global mean.

    # ---- Global mean for all asset parameters (learned) ----
    global_param_loc = pyro.param(
        "global_param_loc", torch.zeros(param_dim, device=device)
    )

    # ---- Low-rank shared factor (learned): captures dependence/covariance structure ---
    #   Shape: [param_dim, r]
    factor = pyro.param(
        "global_param_factor", torch.randn(param_dim, r, device=device) * 0.01
    )

    # ---- Diagonal variance (learned): ensures full covariance is positive definite ----
    diag = pyro.param(
        "global_param_diag",
        torch.ones(param_dim, device=device),
        constraint=constraints.positive,
    )

    # ---- Asset-specific mean offsets (learned) ----
    # Each asset's latent mean = global_param_loc + offset[i]
    asset_offset = pyro.param(
        "local_param_offset", torch.zeros(N_assets, param_dim, device=device)
    )
    # Compute per-asset mean for their latent parameter vector:
    asset_param_mean = global_param_loc + asset_offset  # Shape: (N_assets, 6)

    # ==================================================================
    # 3. ASSET-SPECIFIC PARAMETER POSTERIOR (IN A PLATE!)
    # ==================================================================
    # Now, sample the posterior for all latent parameter vectors.
    # Use the same global 'factor' and 'diag' for all assets, but each gets a unique mean.

    # Pyro's "plate" allows vectorized sampling over all assets
    with pyro.plate("assets_plate", N_assets, dim=-1, device=device):

        # Sample asset param vectors from a low-rank MVN:
        #   (mean shape: [N_assets, 6], factor: shared, diag: shared)
        z = pyro.sample(
            "param_vector",
            dist.LowRankMultivariateNormal(
                loc=asset_param_mean,
                cov_factor=factor.expand(
                    N_assets, -1, -1
                ),  # [N_assets, param_dim, r]
                cov_diag=diag.expand(N_assets, -1),  # [N_assets, param_dim]
            ),
        )
        # should show ([N_assets], param_dim)
        print("plate axis - assets, z shape:", z.shape)

        # ---------------------- Parameter Transformations ------------------------
        #   Rescale latent Gaussian variables into their actual support:
        #   (e.g., omega > 0, ab_sum in (0,1), etc)
        omega_val = torch.exp(z[..., 0])  # positive (GARCH omega parameter)
        ab_sum_val = torch.sigmoid(
            z[..., 1]
        )  # (0,1) via sigmoid (for alpha+beta sum)
        ab_frac_val = torch.sigmoid(
            z[..., 2]
        )  # (0,1) via sigmoid (for alpha fraction)
        phi_val = z[..., 3]  # AR(1) coefficient: real
        sigma0_val = torch.exp(
            z[..., 4]
        )  # initial conditional volatility: positive
        nu_m2_val = torch.exp(z[..., 5])  # degrees-of-freedom-2: positive

        # ---------------------- Asset-level Point Values -------------------------
        # Sample remaining asset parameters (in a point mass way, for SVI):
        pyro.sample("omega", dist.Delta(omega_val))
        pyro.sample("alpha_beta_sum", dist.Delta(ab_sum_val))
        pyro.sample("alpha_frac", dist.Delta(ab_frac_val))
        pyro.sample("phi", dist.Delta(phi_val))
        pyro.sample("sigma0", dist.Delta(sigma0_val))
        pyro.sample("nu_minus2", dist.Delta(nu_m2_val))


if __name__ == "__main__":
    # Dummy data:# [T, N]
    dummy_returns = torch.zeros(3, 2)

    pyro.render_model(
        ar_garch2_studentt_guide,
        model_args=(dummy_returns,),
        render_params=True,
        render_distributions=True,
        filename="guide_02.png",
    )
