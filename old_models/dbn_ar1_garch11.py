import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints


def generate_and_transform_forex_data(
    W=2, days_per_week=1, hours_per_day=4, df=5.0, loc=0.0, scale=0.001, random_seed=42
):
    """
    Generate a dummy dataset of hourly returns for a forex-like asset using a
    StudentT distribution (heavier tails). Then transform these hourly returns
    into daily and hourly PyTorch tensors.

    Arguments:
      W: Number of full trading weeks to generate.
      days_per_week: Number of trading days per week (usually 5).
      hours_per_day: Number of hourly blocks per day.
      df: Degrees of freedom for the StudentT distribution.
      loc: Mean (location) of the distribution.
      scale: Scale parameter of the distribution.
      random_seed: Random seed for reproducibility.

    Returns:
      daily_return: A torch.Tensor of shape [W, days_per_week],
                    containing the sum of hourly returns per day.
      hourly_return: A torch.Tensor of shape [W, days_per_week, hours_per_day],
                     containing the raw hourly returns.
      df_out: A pandas DataFrame with the single column "return", indexed by hour.
    """
    # 1) Set seed and compute the total number of hours
    np.random.seed(random_seed)
    N = W * days_per_week * hours_per_day

    # 2) Sample from a StudentT distribution via Pyro
    with torch.no_grad():
        t_samples = dist.StudentT(df, loc=loc, scale=scale).sample((N,))
    t_samples = t_samples.numpy()

    # 3) Create an hourly date range for these N data points
    start_datetime = pd.Timestamp("2023-01-01 00:00:00")
    idx = pd.date_range(start=start_datetime, periods=N, freq="h")

    # 4) Build the DataFrame
    df_out = pd.DataFrame({"return": t_samples}, index=idx)

    # 5) Reshape into [W, days_per_week, hours_per_day]
    hourly_data = t_samples.reshape(W, days_per_week, hours_per_day)

    # 6) Construct daily returns by summing across hours
    daily_sums = hourly_data.sum(axis=2)  # shape [W, days_per_week]

    # 7) Convert to PyTorch tensors
    daily_return = torch.tensor(daily_sums, dtype=torch.float)
    hourly_return = torch.tensor(hourly_data, dtype=torch.float)

    # 8) Return both tensors and the original DataFrame
    return daily_return, hourly_return


# Yes, precisely. Instead of having a single random variable weekly_sigma (drawn from an Exponential)
# that is constant across all weeks, you would have a time-varying sequence of std deviations
# [sigma_0, sigma_1, ..., sigma_{W−1}] driven by a GARCH-like update. Then, inside the model’s AR(1) step:

# weekly_latent_w ~ Normal(phi × weekly_latent_{w−1}, sigma_w)

# you replace sigma_w with the output of your GARCH or Stochastic Volatility process at week w.
# Conceptually, you are saying:
# • Each week’s “shock” to the latent factor now depends on the prior week’s variance or residual, rather than being a single fixed parameter drawn from an Exponential.
# • The GARCH mechanism can capture volatility clustering over weeks (some weeks are “high variance,” others “low variance,” and they tend to cluster).


def model(
    daily_return,
    hourly_return,
    W=2,
    trading_days=1,
    prior_mean=0.0,
    prior_std=1.0,
):
    """
    A Pyro model capturing a weekly latent factor that drives daily and hourly returns,
    with GARCH(1,1) for the weekly volatility.

    Overview (only change is item #1 below):
      1) weekly_sigma_w is now governed by a GARCH(1,1) recursion:
         sigma_w^2 = omega + alpha * e_{w-1}^2 + beta * sigma_{w-1}^2
         where e_{w-1} = weekly_latent_{w-1} - phi * weekly_latent_{w-2}.
         For w=0 we treat the variance as an initial random draw, then apply recursion.

      2) AR(1) process for the latent factor:
         weekly_latent_w ~ Normal(phi * weekly_latent_{w-1}, sigma_w).

      3) Observations (daily and hourly returns) are modeled via StudentT distributions:
         daily:  StudentT(df_daily, loc=weekly_latent_w, scale=daily_obs_scale)
         hourly: StudentT(df_hourly, loc=weekly_latent_w + hourly_offset, scale=hourly_obs_scale)
         The StudentT distribution helps model heavier tails often seen in financial returns.

      4) An hourly_offset term captures intraday variation around the weekly latent factor.

    Args:
        daily_return (torch.Tensor):
            Observed daily returns, shaped [W, trading_days].
        hourly_return (torch.Tensor):
            Observed hourly returns, shaped [W, trading_days, number_of_hours].
        W (int):
            Number of weeks to model.
        trading_days (int):
            Number of trading days per week. Often 5 for Monday-Friday.
        prior_mean (float):
            Prior mean for the first week's latent factor.
        prior_std (float):
            Prior standard deviation for the first week's latent factor.

    Random Variables (key changes):
        garch_omega, garch_alpha, garch_beta: GARCH(1,1) parameters.
        garch_sigma_init: Initial volatility for the first step in the GARCH recursion.
        weekly_latent_w: Latent factor for week w, with time-varying sigma_w.
    """

    # -----------------------------
    # 1) GARCH(1,1) parameters
    # -----------------------------
    garch_omega = pyro.sample("garch_omega", dist.Exponential(torch.tensor(1.0)))
    garch_alpha = pyro.sample("garch_alpha", dist.Exponential(torch.tensor(1.0)))
    garch_beta = pyro.sample("garch_beta", dist.Exponential(torch.tensor(1.0)))

    # This is our initial weekly volatility: sigma_0
    # (We will use recursion for subsequent weeks)
    sigma_prev = pyro.sample("garch_sigma_init", dist.Exponential(torch.tensor(1.0)))

    # -----------------------------
    # 2) StudentT distribution parameters for daily and hourly returns
    # -----------------------------
    df_daily = pyro.sample("df_daily", dist.Exponential(torch.tensor(0.5)))
    df_hourly = pyro.sample("df_hourly", dist.Exponential(torch.tensor(0.5)))
    daily_obs_scale = pyro.sample("daily_obs_scale", dist.Exponential(torch.tensor(1.0)))
    hourly_obs_scale = pyro.sample("hourly_obs_scale", dist.Exponential(torch.tensor(1.0)))

    # -----------------------------
    # 3) Hourly offsets for intraday variation
    # -----------------------------
    H = hourly_return.shape[-1]
    offset_scale = pyro.sample("offset_scale", dist.Exponential(torch.tensor(1.0)))
    hourly_offset = pyro.sample(
        "hourly_offset",
        dist.Normal(torch.zeros(H), offset_scale * torch.ones(H)),
    )

    # AR(1) coefficient
    phi = pyro.sample("phi", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))

    # Keep track of previous week's latent factor
    weekly_latent_prev = torch.tensor(prior_mean)

    # For the GARCH recursion, we also keep track of the "residual" e_{w-1}.
    # Initialize e_prev so that the recursion can start properly at w=1.
    # For the first week, we will set e_0 = weekly_latent_0 - prior_mean after sampling week 0.
    e_prev = torch.tensor(0.0)

    # -----------------------------
    # 4) Loop over weeks with AR(1) + GARCH(1,1)
    # -----------------------------
    for w_idx in pyro.markov(range(W)):

        if w_idx == 0:
            # First week: use prior mean & std for the distribution
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev, prior_std),
            )
            # Residual e_0 = (latent_0 - prior_mean). We set it for use in next step
            e_prev = weekly_latent_w - weekly_latent_prev
        else:
            # GARCH(1,1) recursion for variance
            # sigma_current^2 = omega + alpha*e_{w-1}^2 + beta*sigma_{w-1}^2
            sigma_current_sq = (
                garch_omega + garch_alpha * (e_prev**2) + garch_beta * (sigma_prev**2)
            )
            sigma_current = torch.sqrt(sigma_current_sq)

            # AR(1) update for the weekly latent factor
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(phi * weekly_latent_prev, sigma_current),
            )

            # Update for next iteration
            e_prev = weekly_latent_w - (phi * weekly_latent_prev)
            sigma_prev = sigma_current

        # 5) Condition daily and hourly observations on weekly_latent_w
        for d_idx in range(trading_days):
            pyro.sample(
                f"daily_obs_{w_idx}_{d_idx}",
                dist.StudentT(df_daily, weekly_latent_w, daily_obs_scale),
                obs=daily_return[w_idx, d_idx],
            )
            for h_idx in range(H):
                pyro.sample(
                    f"hourly_obs_{w_idx}_{d_idx}_{h_idx}",
                    dist.StudentT(
                        df_hourly,
                        weekly_latent_w + hourly_offset[h_idx],
                        hourly_obs_scale,
                    ),
                    obs=hourly_return[w_idx, d_idx, h_idx],
                )

        weekly_latent_prev = weekly_latent_w


def guide(
    daily_return,
    hourly_return,
    W=2,
    trading_days=1,
    prior_mean=0.0,
    prior_std=1.0,
):
    """
    Variational Guide for the AR(1) weekly latent factor model with GARCH(1,1).

    Overview (change is item #1 below):
      1) Instead of a single 'weekly_sigma', we define variational distributions
         for garch_omega, garch_alpha, garch_beta, and garch_sigma_init.

      2) We keep the strategy of a MultivariateNormal for weekly_latent_{0..W-1}
         to allow correlations among the latent factors.

      3) The rest of the parameters (df_daily, df_hourly, offset_scale, etc.)
         remain as in the original guide, each with a mean-field approach.
    """

    # -----------------------------
    # 1) GARCH(1,1) parameters
    # -----------------------------
    # omega
    log_omega_loc = pyro.param("garch_omega_loc", torch.tensor(0.0))
    log_omega_scale = pyro.param(
        "garch_omega_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_omega", dist.LogNormal(log_omega_loc, log_omega_scale))

    # alpha
    log_alpha_loc = pyro.param("garch_alpha_loc", torch.tensor(0.0))
    log_alpha_scale = pyro.param(
        "garch_alpha_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_alpha", dist.LogNormal(log_alpha_loc, log_alpha_scale))

    # beta
    log_beta_loc = pyro.param("garch_beta_loc", torch.tensor(0.0))
    log_beta_scale = pyro.param(
        "garch_beta_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_beta", dist.LogNormal(log_beta_loc, log_beta_scale))

    # initial volatility sigma_0
    log_sigma_init_loc = pyro.param("garch_sigma_init_loc", torch.tensor(0.0))
    log_sigma_init_scale = pyro.param(
        "garch_sigma_init_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_sigma_init", dist.LogNormal(log_sigma_init_loc, log_sigma_init_scale))

    # -----------------------------
    # 2) Degrees of freedom for StudentT
    # -----------------------------
    log_df_daily_loc = pyro.param("log_df_daily_loc", torch.tensor(0.0))
    log_df_daily_scale = pyro.param(
        "log_df_daily_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("df_daily", dist.LogNormal(log_df_daily_loc, log_df_daily_scale))

    log_df_hourly_loc = pyro.param("log_df_hourly_loc", torch.tensor(0.0))
    log_df_hourly_scale = pyro.param(
        "log_df_hourly_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("df_hourly", dist.LogNormal(log_df_hourly_loc, log_df_hourly_scale))

    # -----------------------------
    # 3) Scales for the StudentT distributions
    # -----------------------------
    log_daily_obs_scale_loc = pyro.param("log_daily_obs_scale_loc", torch.tensor(0.0))
    log_daily_obs_scale_scale = pyro.param(
        "log_daily_obs_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "daily_obs_scale",
        dist.LogNormal(log_daily_obs_scale_loc, log_daily_obs_scale_scale),
    )

    log_hourly_obs_scale_loc = pyro.param("log_hourly_obs_scale_loc", torch.tensor(0.0))
    log_hourly_obs_scale_scale = pyro.param(
        "log_hourly_obs_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "hourly_obs_scale",
        dist.LogNormal(log_hourly_obs_scale_loc, log_hourly_obs_scale_scale),
    )

    # -----------------------------
    # 4) Offset scale and hourly offsets
    # -----------------------------
    log_offset_scale_loc = pyro.param("log_offset_scale_loc", torch.tensor(0.0))
    log_offset_scale_scale = pyro.param(
        "log_offset_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("offset_scale", dist.LogNormal(log_offset_scale_loc, log_offset_scale_scale))

    H = hourly_return.shape[-1]
    hourly_offset_loc = pyro.param("hourly_offset_loc", torch.zeros(H))
    hourly_offset_scale = pyro.param(
        "hourly_offset_scale", 0.1 * torch.ones(H), constraint=constraints.positive
    )
    pyro.sample("hourly_offset", dist.Normal(hourly_offset_loc, hourly_offset_scale))

    # -----------------------------
    # 5) AR(1) coefficient phi
    # -----------------------------
    phi_loc = pyro.param("phi_loc", torch.tensor(0.0))
    phi_scale = pyro.param("phi_scale", torch.tensor(0.1), constraint=constraints.positive)
    pyro.sample("phi", dist.Normal(phi_loc, phi_scale))

    # -----------------------------
    # 6) Correlated weekly latents
    # -----------------------------
    weekly_latent_loc = pyro.param("weekly_latent_loc", torch.zeros(W))
    weekly_latent_raw_tril = pyro.param(
        "weekly_latent_raw_tril",
        0.05 * torch.eye(W),
        constraint=constraints.lower_cholesky,
    )
    weekly_latent_scale_tril = torch.tril(weekly_latent_raw_tril)

    all_weeks = pyro.sample(
        "weekly_latent_all",
        dist.MultivariateNormal(weekly_latent_loc, scale_tril=weekly_latent_scale_tril),
    )

    # Tie each weekly_latent_{w_idx} to the relevant element of all_weeks
    for w_idx in pyro.markov(range(W)):
        pyro.sample(f"weekly_latent_{w_idx}", dist.Delta(all_weeks[w_idx]))


if __name__ == "__main__":
    daily_return, hourly_return = generate_and_transform_forex_data()

    pyro.render_model(
        model,
        model_args=(daily_return, hourly_return),
        filename="model_dbn_ar1_garch11.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )

    pyro.render_model(
        guide,
        model_args=(daily_return, hourly_return),
        filename="guide_dbn_ar1_garch11.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
