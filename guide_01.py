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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

# ====== Parameter initial values (define at module scope, outside guide/model) ======
OMEGA_LOC_INIT = torch.full((), 0.0, device=DEVICE)
OMEGA_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
AB_SUM_A_INIT = torch.full((), 2.0, device=DEVICE)
AB_SUM_B_INIT = torch.full((), 2.0, device=DEVICE)
AB_FRAC_A_INIT = torch.full((), 2.0, device=DEVICE)
AB_FRAC_B_INIT = torch.full((), 2.0, device=DEVICE)
PHI_LOC_INIT = torch.full((), 0.0, device=DEVICE)
PHI_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
SIGMA0_LOC_INIT = torch.full((), 0.0, device=DEVICE)
SIGMA0_SCALE_INIT = torch.full((), 0.1, device=DEVICE)
NU_M2_LOC_INIT = torch.full((), 1.0, device=DEVICE)
NU_M2_SCALE_INIT = torch.full((), 0.1, device=DEVICE)


def ar_garch1_studentt_guide(*args, **kwargs):
    device = kwargs.get("device", "cpu")
    # print(f"\nGUIDE DEBUG - device:{device}")
    # omega: GARCH intercept (>0)
    omega_loc = pyro.param("omega_loc", OMEGA_LOC_INIT.clone())
    omega_scale = pyro.param(
        "omega_scale",
        OMEGA_SCALE_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # alpha_beta_sum: stationarity constraint (0,1)
    ab_sum_a = pyro.param(
        "ab_sum_a",
        AB_SUM_A_INIT.clone(),
        constraint=dist.constraints.positive,
    )
    ab_sum_b = pyro.param(
        "ab_sum_b",
        AB_SUM_B_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # alpha_frac: (0,1)
    ab_frac_a = pyro.param(
        "ab_frac_a",
        AB_FRAC_A_INIT.clone(),
        constraint=dist.constraints.positive,
    )
    ab_frac_b = pyro.param(
        "ab_frac_b",
        AB_FRAC_B_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # phi: AR(1) coefficient (real-valued)
    phi_loc = pyro.param("phi_loc", PHI_LOC_INIT.clone())
    phi_scale = pyro.param(
        "phi_scale",
        PHI_SCALE_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # sigma0: initial conditional stdev (>0)
    sigma0_loc = pyro.param("sigma0_loc", SIGMA0_LOC_INIT.clone())
    sigma0_scale = pyro.param(
        "sigma0_scale",
        SIGMA0_SCALE_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # nu_minus2: for StudentT (>0; model shifts by +2)
    nu_m2_loc = pyro.param("nu_m2_loc", NU_M2_LOC_INIT.clone())
    nu_m2_scale = pyro.param(
        "nu_m2_scale",
        NU_M2_SCALE_INIT.clone(),
        constraint=dist.constraints.positive,
    )

    # Guide distributions:
    pyro.sample("omega", dist.LogNormal(omega_loc, omega_scale))
    pyro.sample("alpha_beta_sum", dist.Beta(ab_sum_a, ab_sum_b))
    pyro.sample("alpha_frac", dist.Beta(ab_frac_a, ab_frac_b))
    pyro.sample("phi", dist.Normal(phi_loc, phi_scale))
    pyro.sample("sigma0", dist.LogNormal(sigma0_loc, sigma0_scale))
    pyro.sample("nu_minus2", dist.LogNormal(nu_m2_loc, nu_m2_scale))


if __name__ == "__main__":
    # Create some dummy data for illustrative purposes (T=5 timesteps)
    dummy_returns = torch.randn(3)

    pyro.render_model(
        ar_garch1_studentt_guide,
        model_args=(dummy_returns,),
        filename="guide_01.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
