# pylint: disable=trailing-whitespace
import pyro
import pyro.distributions as dist
import torch


def guide(
    returns,              # [batch_size, max_T]
    lengths,              # [batch_size]
    args,                 # Dummy, just for API compatibility
    prior_predictive_checks: bool = False,
    device=torch.device("cpu"),
):
    returns = returns.to(device)
    lengths = lengths.to(device)
    batch_size, max_T = returns.shape

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
    # Variational parameters are registered globally for all assets
    loc = pyro.param("local_loc")           # shape: [num_assets, per_asset_param_dim]
    scale_tril = pyro.param(
        "local_scale_tril"
    )                                       # shape: [num_assets, per_asset_param_dim, per_asset_param_dim]

    # Sample only the part of "loc" and "scale_tril" for the current batch
    with pyro.plate("assets", batch_size, dim=-2):
        local_loc = loc[:batch_size, :]                      # Slice the first batch_size assets for current batch
        local_scale_tril = scale_tril[:batch_size, :, :]     # Same here
        local_latents = pyro.sample(
            "local_latents",
            dist.MultivariateNormal(local_loc, scale_tril=local_scale_tril).to_event(1)
        )
