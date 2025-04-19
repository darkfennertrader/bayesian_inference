# pylint: disable=trailing-whitespace
import pyro
import pyro.distributions as dist
import torch


def guide(
    returns,              # [num_assets, max_T]
    lengths,              # [num_assets]
    args,                 # Dummy, just for API compatibility
    prior_predictive_checks: bool = False,
    device=torch.device("cpu"),
    batch_size=None
):
    returns = returns.to(device)
    lengths = lengths.to(device)
    num_assets, max_T = returns.shape

    # ---- MEAN-FIELD FOR GLOBALS ----------
    # Each uses unconstrained loc/scale for variational params
    omega_mu_loc = pyro.param("omega_mu_loc", torch.tensor(1.5, device=device))
    omega_mu_scale = pyro.param("omega_mu_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    omega_mu = pyro.sample("omega_mu", dist.Normal(omega_mu_loc, omega_mu_scale))
      
    omega_sigma_loc = pyro.param("omega_sigma_loc", torch.tensor(1.5, device=device))
    omega_sigma_scale = pyro.param("omega_sigma_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    omega_sigma = pyro.sample("omega_sigma", dist.Normal(omega_sigma_loc, omega_sigma_scale))
    
    ab_sum_a_hyper_loc = pyro.param("ab_sum_a_hyper_loc", torch.tensor(2.0, device=device))
    ab_sum_a_hyper_scale = pyro.param("ab_sum_a_hyper_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    ab_sum_a_hyper = pyro.sample("ab_sum_a_hyper", dist.Normal(ab_sum_a_hyper_loc, ab_sum_a_hyper_scale))
    
    ab_sum_b_hyper_loc = pyro.param("ab_sum_b_hyper_loc", torch.tensor(2.0, device=device))
    ab_sum_b_hyper_scale = pyro.param("ab_sum_b_hyper_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    ab_sum_b_hyper = pyro.sample("ab_sum_b_hyper", dist.Normal(ab_sum_b_hyper_loc, ab_sum_b_hyper_scale))
    
    ab_frac_a_hyper_loc = pyro.param("ab_frac_a_hyper_loc", torch.tensor(2.0, device=device))
    ab_frac_a_hyper_scale = pyro.param("ab_frac_a_hyper_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    ab_frac_a_hyper = pyro.sample("ab_frac_a_hyper", dist.Normal(ab_frac_a_hyper_loc, ab_frac_a_hyper_scale))
    
    ab_frac_b_hyper_loc = pyro.param("ab_frac_b_hyper_loc", torch.tensor(2.0, device=device))
    ab_frac_b_hyper_scale = pyro.param("ab_frac_b_hyper_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    ab_frac_b_hyper = pyro.sample("ab_frac_b_hyper", dist.Normal(ab_frac_b_hyper_loc, ab_frac_b_hyper_scale))

    phi_mu_loc = pyro.param("phi_mu_loc", torch.tensor(0.0, device=device))
    phi_mu_scale = pyro.param("phi_mu_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    phi_mu = pyro.sample("phi_mu", dist.Normal(phi_mu_loc, phi_mu_scale))
    
    phi_sigma_loc = pyro.param("phi_sigma_loc", torch.tensor(1.0, device=device))
    phi_sigma_scale = pyro.param("phi_sigma_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    phi_sigma = pyro.sample("phi_sigma", dist.Normal(phi_sigma_loc, phi_sigma_scale))
    
    sigma_init_mu_loc = pyro.param("sigma_init_mu_loc", torch.tensor(10.0, device=device))
    sigma_init_mu_scale = pyro.param("sigma_init_mu_scale", torch.tensor(1.0, device=device), constraint=dist.constraints.positive)
    sigma_init_mu = pyro.sample("sigma_init_mu", dist.Normal(sigma_init_mu_loc, sigma_init_mu_scale))
    
    sigma_init_sigma_loc = pyro.param("sigma_init_sigma_loc", torch.tensor(10.0, device=device))
    sigma_init_sigma_scale = pyro.param("sigma_init_sigma_scale", torch.tensor(1.0, device=device), constraint=dist.constraints.positive)
    sigma_init_sigma = pyro.sample("sigma_init_sigma", dist.Normal(sigma_init_sigma_loc, sigma_init_sigma_scale))
    
    df_mu_loc = pyro.param("df_mu_loc", torch.tensor(10.0, device=device))
    df_mu_scale = pyro.param("df_mu_scale", torch.tensor(1.0, device=device), constraint=dist.constraints.positive)
    df_mu = pyro.sample("df_mu", dist.Normal(df_mu_loc, df_mu_scale))
    
    df_sigma_loc = pyro.param("df_sigma_loc", torch.tensor(1.0, device=device))
    df_sigma_scale = pyro.param("df_sigma_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    df_sigma = pyro.sample("df_sigma", dist.Normal(df_sigma_loc, df_sigma_scale))
    
    lambda_decay_alpha = pyro.param("lambda_decay_alpha", torch.tensor(2.0, device=device), constraint=dist.constraints.positive)
    lambda_decay_beta  = pyro.param("lambda_decay_beta", torch.tensor(2.0, device=device), constraint=dist.constraints.positive)
    lambda_decay = pyro.sample("lambda_decay", dist.Beta(lambda_decay_alpha, lambda_decay_beta))

    # ---- STRUCTURED MULTIVARIATE FOR LOCALS (PER ASSET) --------
    # Block: per asset, ALL local params enter a single multivariate Normal
    per_asset_param_dim = 8
    # Order: garch_omega, alpha_beta_sum, alpha_frac, phi, garch_sigma_init, degrees_of_freedom, asset_weight, (possible npad for alignment)
    with pyro.plate("assets", num_assets, dim=-2):
        loc = pyro.param("local_loc", torch.zeros(num_assets, per_asset_param_dim, device=device))
        scale_tril = pyro.param(
            "local_scale_tril",
            torch.stack([torch.eye(per_asset_param_dim, device=device) for _ in range(num_assets)]), 
            constraint=dist.constraints.lower_cholesky
        )
        local_latents = pyro.sample(
            "local_latents",
            dist.MultivariateNormal(loc, scale_tril=scale_tril).to_event(1)
        )

    # You will then decode each latent vector into its respective parameter:
    #  garch_omega_i = local_latents[:, 0]
    #  alpha_beta_sum_i = local_latents[:, 1]
    #  alpha_frac_i = local_latents[:, 2]
    #  phi_i = local_latents[:, 3]
    #  garch_sigma_init_i = local_latents[:, 4]
    #  degrees_of_freedom_i = local_latents[:, 5]
    #  asset_weight_i = local_latents[:, 6]
    #  [Slot 7 may be left as npad/unused or you can add another parameter]

    # If obs masking or batch mode is needed adjust accordingly.
