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


def ar_garch_model_student_t_multi_asset_partial_pooling(
    returns, lengths, **kwargs
):
    """
    Parallel AR(1)-GARCH(1,1) model with Student-T innovations and per-asset partial-pooling.
    Parameter packing order (asset vector):
      0: garch_omega (LogNormal)
      1: alpha_beta_sum (Beta)
      2: alpha_frac (Beta)
      3: phi (Normal)
      4: garch_sigma_init (LogNormal)
      5: degrees_of_freedom (>2, LogNormal + 2)
      6: asset_weight (Beta)
    """
    args = kwargs.get("args", {})
    prior_predictive_checks = kwargs.get("prior_predictive_checks", False)
    device = kwargs.get("device", torch.device("cpu"))
    indices = kwargs.get("indices", None)
    assert indices is not None, "Indices must be passed to the guide!"

    returns = returns.to(device)
    lengths = lengths.to(device)
    ################################################################
    print(
        f"MODEL() DEBUG: returns.device: {returns.device}, lengths.device: {lengths.device}, device set: {device}"
    )
    ###############################################################

    batch_size, max_T = returns.shape

    ################################################################
    print(
        "MODEL() | returns.shape:",
        returns.shape,
        "| lengths.shape:",
        lengths.shape,
        "| batch_size:",
        batch_size,
        "| max_T:",
        max_T,
    )
    ##################################################################
    # ==== GLOBAL HYPERPRIORS ====
    omega_mu = pyro.sample(
        "omega_mu",
        dist.LogNormal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        ),
    )
    omega_sigma = pyro.sample(
        "omega_sigma",
        dist.LogNormal(
            torch.tensor(0.0, device=device), torch.tensor(0.5, device=device)
        ),
    )

    ab_sum_a_hyper = pyro.sample(
        "ab_sum_a_hyper",
        dist.LogNormal(
            torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)
        ),
    )
    ab_sum_b_hyper = pyro.sample(
        "ab_sum_b_hyper",
        dist.LogNormal(
            torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)
        ),
    )
    ab_frac_a_hyper = pyro.sample(
        "ab_frac_a_hyper",
        dist.LogNormal(
            torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)
        ),
    )
    ab_frac_b_hyper = pyro.sample(
        "ab_frac_b_hyper",
        dist.LogNormal(
            torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)
        ),
    )
    phi_mu = pyro.sample(
        "phi_mu",
        dist.Normal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        ),
    )
    phi_sigma = pyro.sample(
        "phi_sigma",
        dist.LogNormal(
            torch.tensor(0.0, device=device), torch.tensor(0.5, device=device)
        ),
    )
    sigma_init_mu = pyro.sample(
        "sigma_init_mu",
        dist.LogNormal(
            torch.tensor(2.0, device=device), torch.tensor(0.5, device=device)
        ),
    )
    sigma_init_sigma = pyro.sample(
        "sigma_init_sigma",
        dist.LogNormal(
            torch.tensor(0.0, device=device), torch.tensor(0.5, device=device)
        ),
    )
    df_mu = pyro.sample(
        "df_mu",
        dist.LogNormal(
            torch.tensor(1.0, device=device), torch.tensor(0.5, device=device)
        ),
    )
    df_sigma = pyro.sample(
        "df_sigma",
        dist.LogNormal(
            torch.tensor(0.0, device=device), torch.tensor(0.5, device=device)
        ),
    )
    lambda_decay = pyro.sample(
        "lambda_decay",
        dist.Beta(
            torch.tensor(2.0, device=device), torch.tensor(2.0, device=device)
        ),
    )

    ###################################################################
    print("MODEL() | about to enter assets plate, batch_size =", batch_size)
    print("MODEL() DEBUG (ASSET PLATE): device", device)
    ######################################################################

    # ==== ASSET-SPECIFIC PARAMETERS ====
    with pyro.plate("assets", batch_size, dim=-2, device=device):
        ######################################################
        print(
            f"MODEL() DEBUG: omega_mu.device={omega_mu.device}, omega_sigma.device={omega_sigma.device}, batch_size={batch_size}"
        )
        #######################################################
        garch_omega = pyro.sample(
            "garch_omega", dist.LogNormal(omega_mu, omega_sigma)
        )
        #########################################################
        print("MODEL DEBUG: garch_omega.device", garch_omega.device)
        #########################################################
        alpha_beta_sum = pyro.sample(
            "alpha_beta_sum", dist.Beta(ab_sum_a_hyper, ab_sum_b_hyper)
        )
        #########################################################
        print("MODEL DEBUG: alpha_beta_sum.device", alpha_beta_sum.device)
        #########################################################
        alpha_frac = pyro.sample(
            "alpha_frac", dist.Beta(ab_frac_a_hyper, ab_frac_b_hyper)
        )
        #########################################################
        print("MODEL DEBUG: alpha_frac.device", alpha_frac.device)
        #########################################################
        phi = pyro.sample("phi", dist.Normal(phi_mu, phi_sigma))
        #########################################################
        print("MODEL DEBUG: phi.device", phi.device)
        #########################################################
        garch_sigma_init = pyro.sample(
            "garch_sigma_init", dist.LogNormal(sigma_init_mu, sigma_init_sigma)
        )
        #########################################################
        print("MODEL DEBUG: garch_sigma_init.device", garch_sigma_init.device)
        #########################################################
        raw_df = pyro.sample(
            "degrees_of_freedom_raw", dist.LogNormal(df_mu, df_sigma)
        )
        #########################################################
        print("MODEL DEBUG: raw_df.device", raw_df.device)
        #########################################################
        degrees_of_freedom = raw_df + 2.0
        asset_weight = pyro.sample(
            "asset_weight",
            dist.Beta(
                torch.tensor(2.0, device=device),
                torch.tensor(2.0, device=device),
            ),
        )
        #########################################################
        print(
            "MODEL DEBUG: degrees_of_freedom.device", degrees_of_freedom.device
        )
        #########################################################

        #####################################################################3
        print("MODEL() | garch_omega.shape", garch_omega.shape)
        print("MODEL() | alpha_beta_sum.shape", alpha_beta_sum.shape)
        print("MODEL() | alpha_frac.shape", alpha_frac.shape)
        print("MODEL() | phi.shape", phi.shape)
        print("MODEL() | garch_sigma_init.shape", garch_sigma_init.shape)
        print("MODEL() | degrees_of_freedom_raw.shape", raw_df.shape)
        print("MODEL() | asset_weight.shape", asset_weight.shape)
        ##################################################################3

        sigma_prev = garch_sigma_init
        e_prev = torch.zeros(batch_size, device=device)
        r_prev = torch.zeros(batch_size, device=device)
        asset_lengths = lengths
        asset_returns = returns

        for t in pyro.markov(
            range(
                max_T
                if getattr(args, "jit", False)
                else asset_lengths.max().item()
            )
        ):
            valid_mask = t < asset_lengths
            decay_exponent = max_T - t - 1

            garch_alpha = alpha_beta_sum * alpha_frac
            garch_beta = alpha_beta_sum * (1.0 - alpha_frac)
            if t == 0:
                sigma_t = sigma_prev
                mean_t = torch.zeros_like(phi)
            else:
                sigma_t = (
                    garch_omega
                    + garch_alpha * (e_prev**2)
                    + garch_beta * (sigma_prev**2)
                ).sqrt()
                mean_t = phi * r_prev

            obs = None if prior_predictive_checks else asset_returns[:, t]
            if obs is not None:
                log_prob = dist.StudentT(
                    degrees_of_freedom, mean_t, sigma_t
                ).log_prob(obs)
                combined_weight = asset_weight * (lambda_decay**decay_exponent)
                weighted_log_prob = torch.where(
                    valid_mask,
                    combined_weight * log_prob,
                    torch.zeros_like(log_prob),
                )
                ############################################################
                print(
                    f"MODEL() weighted_log_prob.device: {weighted_log_prob.device}, combined_weight.device: {combined_weight.device}, log_prob.device: {log_prob.device}"
                )
                #############################################################

                pyro.factor(f"weighted_decay_{t}", weighted_log_prob)
            with poutine.mask(mask=valid_mask):
                r_t = pyro.sample(
                    f"r_{t}",
                    dist.StudentT(degrees_of_freedom, mean_t, sigma_t),
                    obs=obs,
                )
                e_prev = r_t - mean_t
                sigma_prev = sigma_t
                r_prev = r_t


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
        model_args=(dummy_returns, dummy_lengths),
        model_kwargs={"args": args},
        filename="ar_garch_StudT_multiassets_partialpool.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    ).unflatten(stagger=1)
