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
    # Number of hourly observations per day
    H = hourly_return.shape[-1]

    # -----------------------------
    # 1) GARCH(1,1) parameters
    # -----------------------------
    garch_omega = pyro.sample("garch_omega", dist.Exponential(1.0))
    garch_alpha = pyro.sample("garch_alpha", dist.Exponential(1.0))
    garch_beta = pyro.sample("garch_beta", dist.Exponential(1.0))
    sigma_prev = pyro.sample("garch_sigma_init", dist.Exponential(1.0))

    # -----------------------------
    # 2) StudentT parameters
    # -----------------------------
    df_daily = pyro.sample("df_daily", dist.Exponential(0.5))
    df_hourly = pyro.sample("df_hourly", dist.Exponential(0.5))
    daily_obs_scale = pyro.sample("daily_obs_scale", dist.Exponential(1.0))
    hourly_obs_scale = pyro.sample("hourly_obs_scale", dist.Exponential(1.0))

    # -----------------------------
    # 3) Hourly offsets
    # -----------------------------
    offset_scale = pyro.sample("offset_scale", dist.Exponential(1.0))
    hourly_offset = pyro.sample(
        "hourly_offset",
        dist.Normal(torch.zeros(H), offset_scale * torch.ones(H)),
    )

    # -----------------------------
    # 4) AR(1) coefficient
    # -----------------------------
    phi = pyro.sample("phi", dist.Normal(0.0, 1.0))

    # -----------------------------
    # 5) Discrete latent regime variable (2 regimes)
    # -----------------------------
    regime_probs = pyro.sample("regime_probs", dist.Dirichlet(torch.tensor([1.0, 1.0])))
    with pyro.plate("weeks", W):
        regime = pyro.sample(
            "regime", dist.Categorical(regime_probs), infer={"enumerate": "parallel"}
        )

    # Regime-specific volatility multipliers
    regime_volatility = pyro.sample(
        "regime_volatility", dist.LogNormal(torch.zeros(2), torch.ones(2))
    )

    weekly_latent_prev = torch.tensor(prior_mean)
    e_prev = torch.tensor(0.0)

    for w_idx in pyro.markov(range(W)):
        if w_idx == 0:
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev, prior_std),
            )
            e_prev = weekly_latent_w - weekly_latent_prev
        else:
            sigma_current_sq = (
                garch_omega + garch_alpha * (e_prev**2) + garch_beta * (sigma_prev**2)
            )
            sigma_current = torch.sqrt(sigma_current_sq) * regime_volatility[regime[w_idx]]

            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(phi * weekly_latent_prev, sigma_current),
            )

            e_prev = weekly_latent_w - (phi * weekly_latent_prev)
            sigma_prev = sigma_current

        # Observations
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
    trading_days=5,
    prior_mean=0.0,
    prior_std=1.0,
):
    H = hourly_return.shape[-1]

    # GARCH parameters
    garch_omega_loc = pyro.param(
        "garch_omega_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_omega", dist.LogNormal(garch_omega_loc, 0.1))

    garch_alpha_loc = pyro.param(
        "garch_alpha_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_alpha", dist.LogNormal(garch_alpha_loc, 0.1))

    garch_beta_loc = pyro.param(
        "garch_beta_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_beta", dist.LogNormal(garch_beta_loc, 0.1))

    sigma_init_loc = pyro.param(
        "sigma_init_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("garch_sigma_init", dist.LogNormal(sigma_init_loc, 0.1))

    # StudentT parameters
    df_daily_loc = pyro.param("df_daily_loc", torch.tensor(1.0), constraint=constraints.positive)
    pyro.sample("df_daily", dist.LogNormal(df_daily_loc, 0.1))

    df_hourly_loc = pyro.param("df_hourly_loc", torch.tensor(1.0), constraint=constraints.positive)
    pyro.sample("df_hourly", dist.LogNormal(df_hourly_loc, 0.1))

    daily_obs_scale_loc = pyro.param(
        "daily_obs_scale_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("daily_obs_scale", dist.LogNormal(daily_obs_scale_loc, 0.1))

    hourly_obs_scale_loc = pyro.param(
        "hourly_obs_scale_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("hourly_obs_scale", dist.LogNormal(hourly_obs_scale_loc, 0.1))

    # Hourly offsets
    offset_scale_loc = pyro.param(
        "offset_scale_loc", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("offset_scale", dist.LogNormal(offset_scale_loc, 0.1))

    hourly_offset_loc = pyro.param("hourly_offset_loc", torch.zeros(H))
    hourly_offset_scale = pyro.param(
        "hourly_offset_scale", 0.1 * torch.ones(H), constraint=constraints.positive
    )
    pyro.sample("hourly_offset", dist.Normal(hourly_offset_loc, hourly_offset_scale))

    # AR(1) coefficient
    phi_loc = pyro.param("phi_loc", torch.tensor(0.0))
    phi_scale = pyro.param("phi_scale", torch.tensor(0.1), constraint=constraints.positive)
    pyro.sample("phi", dist.Normal(phi_loc, phi_scale))

    # Regime probabilities
    regime_probs_param = pyro.param(
        "regime_probs_param", torch.ones(2), constraint=constraints.simplex
    )
    pyro.sample("regime_probs", dist.Dirichlet(regime_probs_param))

    # Regime volatility
    regime_volatility_loc = pyro.param("regime_volatility_loc", torch.zeros(2))
    regime_volatility_scale = pyro.param(
        "regime_volatility_scale", torch.ones(2), constraint=constraints.positive
    )
    pyro.sample("regime_volatility", dist.LogNormal(regime_volatility_loc, regime_volatility_scale))

    # Weekly latent factors (diagonal covariance)
    weekly_latent_loc = pyro.param("weekly_latent_loc", torch.zeros(W))
    weekly_latent_scale = pyro.param(
        "weekly_latent_scale", 0.1 * torch.ones(W), constraint=constraints.positive
    )
    for w_idx in pyro.markov(range(W)):
        pyro.sample(
            f"weekly_latent_{w_idx}",
            dist.Normal(weekly_latent_loc[w_idx], weekly_latent_scale[w_idx]),
        )


if __name__ == "__main__":
    daily_return, hourly_return = generate_and_transform_forex_data()

    pyro.render_model(
        model,
        model_args=(daily_return, hourly_return),
        filename="model_dbn_ar1_garch11_regimeshift.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )

    pyro.render_model(
        guide,
        model_args=(daily_return, hourly_return),
        filename="guide_dbn_ar1_garch11_regimeshift.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
