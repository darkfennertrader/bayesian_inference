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


def ar_garch2_studentt_model(returns, **kwargs):
    """
    Hierarchical Dynamic Bayesian Network for AR(1)-GARCH(1,1), Student-T innovations.
    Multi-asset: returns shape [T, N_assets]
    """
    print("\nMODEL:")

    # Select computation device and push data to it
    device = kwargs.get("device", "cpu")
    returns = returns.to(device)
    T, N_assets = returns.shape
    print(f"returns shape: Timesteps: {T}, Assets: {N_assets}")

    # ===== GLOBAL HYPERPRIORS =====
    omega_loc = pyro.sample("omega_loc", dist.Normal(0.0, 1.0))
    # print("omega_loc:")
    # print(omega_loc.detach().cpu().numpy())

    omega_scale = pyro.sample("omega_scale", dist.HalfCauchy(1.0))
    # print("omega_scale:")
    # print(omega_scale.detach().cpu().numpy())

    ab_sum_a = pyro.sample("ab_sum_a", dist.Gamma(2.0, 1.0))
    # print("ab_sum_a:")
    # print(ab_sum_a.detach().cpu().numpy())

    ab_sum_b = pyro.sample("ab_sum_b", dist.Gamma(2.0, 1.0))
    # print("ab_sum_b:")
    # print(ab_sum_b.detach().cpu().numpy())

    ab_frac_a = pyro.sample("ab_frac_a", dist.Gamma(2.0, 1.0))
    # print("ab_frac_a:")
    # print(ab_frac_a.detach().cpu().numpy())

    ab_frac_b = pyro.sample("ab_frac_b", dist.Gamma(2.0, 1.0))
    # print("ab_frac_b:")
    # print(ab_frac_b.detach().cpu().numpy())

    phi_loc = pyro.sample("phi_loc", dist.Normal(0.0, 1.0))
    # print("phi_loc:")
    # print(phi_loc.detach().cpu().numpy())

    phi_scale = pyro.sample("phi_scale", dist.HalfCauchy(1.0))
    # print("phi_scale:")
    # print(phi_scale.detach().cpu().numpy())

    sigma0_loc = pyro.sample("sigma0_loc", dist.Normal(0.0, 1.0))
    # print("sigma0_loc:")
    # print(sigma0_loc.detach().cpu().numpy())

    sigma0_scale = pyro.sample("sigma0_scale", dist.HalfCauchy(1.0))
    # print("sigma0_scale:")
    # print(sigma0_scale.detach().cpu().numpy())

    nu_m2_loc = pyro.sample("nu_m2_loc", dist.Normal(1.0, 0.3))
    # print("nu_m2_loc:")
    # print(nu_m2_loc.detach().cpu().numpy())

    nu_m2_scale = pyro.sample("nu_m2_scale", dist.HalfCauchy(0.5))
    # print("nu_m2_scale:")
    # print(nu_m2_scale.detach().cpu().numpy())

    # ===== ASSET-SPECIFIC PRIORS / PARAMETERS =====
    with pyro.plate("assets_plate", N_assets):
        omega = pyro.sample("omega", dist.LogNormal(omega_loc, omega_scale))
        # print("omega:")
        # print(omega.detach().cpu().numpy())

        alpha_beta_sum = pyro.sample(
            "alpha_beta_sum", dist.Beta(ab_sum_a, ab_sum_b)
        )
        # print("alpha_beta_sum:")
        # print(alpha_beta_sum.detach().cpu().numpy())

        alpha_frac = pyro.sample("alpha_frac", dist.Beta(ab_frac_a, ab_frac_b))
        # print("alpha_frac:")
        # print(alpha_frac.detach().cpu().numpy())

        phi = pyro.sample("phi", dist.Normal(phi_loc, phi_scale))
        # print("phi:")
        # print(phi.detach().cpu().numpy())

        sigma0 = pyro.sample("sigma0", dist.LogNormal(sigma0_loc, sigma0_scale))
        # print("sigma0:")
        # print(sigma0.detach().cpu().numpy())

        nu_minus2 = pyro.sample(
            "nu_minus2", dist.LogNormal(nu_m2_loc, nu_m2_scale)
        )
        # print("nu_minus2:")
        # print(nu_minus2.detach().cpu().numpy())

        garch_alpha = alpha_beta_sum * alpha_frac
        # print("garch_alpha:")
        # print(garch_alpha.detach().cpu().numpy())

        garch_beta = alpha_beta_sum * (1.0 - alpha_frac)
        with ignore_jit_warnings():
            assert (garch_alpha >= 0).all(), "garch_alpha negative"
            assert (garch_beta >= 0).all(), "garch_beta negative"
            assert (
                garch_alpha + garch_beta < 1.0
            ).all(), "alpha+beta instability"

        # print("garch_beta:")
        # print(garch_beta.detach().cpu().numpy())

        # ===== STATE INITIALIZATION (time zero) =====
        print()
        sigma_prev = sigma0
        print("sigma_prev:")
        print(sigma_prev.detach().cpu())

        r_prev = torch.zeros(N_assets, device=device)
        print("r_prev:")
        print(r_prev.detach().cpu())

        e_prev = torch.zeros(N_assets, device=device)
        print("e_prev:")
        print(e_prev.detach().cpu())

        print("returns")
        print(returns.detach().cpu())

        # ===== DYNAMIC TIME LOOP =====
        with pyro.markov():
            for t in range(T):
                print(f"\nMARKOV LOOP: {t}")
                # --- GARCH volatility recursion ---
                if t > 0:
                    sigma_t = (
                        omega
                        + garch_alpha * e_prev**2
                        + garch_beta * sigma_prev**2
                    ).sqrt()
                else:
                    sigma_t = sigma_prev
                print(f"sigma_t [{t}]:")
                print(sigma_t.detach().cpu())
                with ignore_jit_warnings():
                    assert not torch.isnan(sigma_t).any(), "sigma_t NaN"

                # --- AR(1) conditional mean ---
                mean_t = (
                    phi * r_prev
                    if t > 0
                    else torch.zeros(N_assets, device=device)
                )
                print(f"mean_t [{t}]:")
                print(mean_t.detach().cpu())
                with ignore_jit_warnings():
                    assert not torch.isnan(mean_t).any(), "mean_t NaN"

                # Only observe/condition if no NaN
                curr_obs = returns[t, :]
                obs_mask = ~torch.isnan(curr_obs)
                curr_obs_obs = curr_obs[obs_mask]  # only observed assets
                print("curr_obs_obs")
                print(curr_obs[obs_mask].detach().cpu())

                pyro.sample(
                    f"r_{t}",
                    dist.StudentT(
                        nu_minus2[obs_mask] + 2.0,
                        mean_t[obs_mask],
                        sigma_t[obs_mask],
                    ),
                    obs=curr_obs_obs,
                )

                # Update state for next step (fill NaNs for all assets)
                r_t_all = torch.empty(N_assets, device=device)
                r_t_all[:] = mean_t
                r_t_all[obs_mask] = curr_obs_obs

                e_prev = r_t_all - mean_t
                print(f"e_prev [{t}]:")
                print(e_prev.detach().cpu())

                r_prev = r_t_all
                print(f"r_prev [{t}]:")
                print(r_prev.detach().cpu())

                sigma_prev = sigma_t


if __name__ == "__main__":
    # Dummy data:# [T, N]
    dummy_returns = torch.zeros(3, 2)

    pyro.render_model(
        ar_garch2_studentt_model,
        model_args=(dummy_returns,),
        filename="model_02.jpg",
        render_params=True,
        render_distributions=True,
    )
