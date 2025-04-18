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


def model(
    daily_return, hourly_return, W=2, trading_days=1, prior_mean=0.0, prior_std=1.0
):
    """
    A Pyro model capturing a week-to-week latent factor that drives daily and hourly returns.
    - The latent factor evolves each week, forming a Markov chain.
    - Each daily and hourly return is observed through a heavy-tailed StudentT distribution.
    - An hourly offset term accounts for intraday variations.

    Args:
        daily_return (torch.Tensor): Observed daily returns shaped [W, trading_days].
        hourly_return (torch.Tensor): Observed hourly returns shaped [W, trading_days, number_of_hours].
        W (int): Number of weeks to model.
        trading_days (int): Number of trading days per week.
        prior_mean (float): Prior mean for the first week's latent factor.
        prior_std (float): Prior standard deviation for the first week's latent factor.
    """

    # 1) Sample the standard deviation for the weekly transition
    #    (how much the latent factor changes from week to week).
    weekly_sigma = pyro.sample("weekly_sigma", dist.Exponential(torch.tensor(1.0)))

    # 2) Sample parameters for the StudentT distributions used for daily and hourly returns:
    #    - df_daily and df_hourly are degrees-of-freedom, controlling tail thickness.
    #    - daily_obs_scale and hourly_obs_scale are scale parameters for daily and hourly returns.
    df_daily = pyro.sample("df_daily", dist.Exponential(torch.tensor(0.5)))
    df_hourly = pyro.sample("df_hourly", dist.Exponential(torch.tensor(0.5)))
    daily_obs_scale = pyro.sample(
        "daily_obs_scale", dist.Exponential(torch.tensor(1.0))
    )
    hourly_obs_scale = pyro.sample(
        "hourly_obs_scale", dist.Exponential(torch.tensor(1.0))
    )

    # 3) Hourly offsets capture intraday variation around the weekly latent factor.
    #    - offset_scale sets the std dev for the offsets.
    #    - hourly_offset is a vector of length H, so each hour can shift differently.
    H = hourly_return.shape[-1]
    offset_scale = pyro.sample("offset_scale", dist.Exponential(torch.tensor(1.0)))
    hourly_offset = pyro.sample(
        "hourly_offset", dist.Normal(torch.zeros(H), offset_scale * torch.ones(H))
    )

    # 4) Keep track of the previous week's latent factor.
    #    For the first week, we use a fixed prior.
    weekly_latent_prev = torch.tensor(prior_mean)

    # 5) Loop over W weeks in a Markov chain.
    for w_idx in pyro.markov(range(W)):

        # If this is the first week, sample a latent factor from a normal distribution
        # centered at prior_mean with std dev prior_std.
        if w_idx == 0:
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev, prior_std),
            )
        else:
            # For subsequent weeks, the latent factor depends on the previous week's
            # factor through a normal distribution with std dev weekly_sigma.
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev, weekly_sigma),
            )

        # Observe the daily and hourly returns given the weekly latent factor
        # using StudentT distributions.
        for d_idx in range(trading_days):

            # Daily observation: StudentT with
            # location = weekly_latent_w
            # scale = daily_obs_scale
            pyro.sample(
                f"daily_obs_{w_idx}_{d_idx}",
                dist.StudentT(df_daily, weekly_latent_w, daily_obs_scale),
                obs=daily_return[w_idx, d_idx],
            )

            # Hourly observations: StudentT with
            # location = weekly_latent_w + hourly_offset[h_idx]
            # scale = hourly_obs_scale
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

        # Update the previous latent with the current week's latent factor
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
    Guide (variational distribution) for weekly_latent_dynamic_model with correlation
    across the weekly latent factors. We still retain a mean-field approach for the
    other parameters, but place a MultivariateNormal distribution on the entire
    [weekly_latent_0, ..., weekly_latent_(W-1)] vector so that SVI can learn correlations.
    """

    # -----------------------------
    # 1) Global scale parameters
    # -----------------------------

    # weekly_sigma > 0 (exponential prior in the model) -> LogNormal in the guide
    log_weekly_sigma_loc = pyro.param("log_weekly_sigma_loc", torch.tensor(0.0))
    log_weekly_sigma_scale = pyro.param(
        "log_weekly_sigma_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "weekly_sigma", dist.LogNormal(log_weekly_sigma_loc, log_weekly_sigma_scale)
    )

    # df_daily > 0
    log_df_daily_loc = pyro.param("log_df_daily_loc", torch.tensor(0.0))
    log_df_daily_scale = pyro.param(
        "log_df_daily_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("df_daily", dist.LogNormal(log_df_daily_loc, log_df_daily_scale))

    # df_hourly > 0
    log_df_hourly_loc = pyro.param("log_df_hourly_loc", torch.tensor(0.0))
    log_df_hourly_scale = pyro.param(
        "log_df_hourly_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("df_hourly", dist.LogNormal(log_df_hourly_loc, log_df_hourly_scale))

    # daily_obs_scale > 0
    log_daily_obs_scale_loc = pyro.param("log_daily_obs_scale_loc", torch.tensor(0.0))
    log_daily_obs_scale_scale = pyro.param(
        "log_daily_obs_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "daily_obs_scale",
        dist.LogNormal(log_daily_obs_scale_loc, log_daily_obs_scale_scale),
    )

    # hourly_obs_scale > 0
    log_hourly_obs_scale_loc = pyro.param("log_hourly_obs_scale_loc", torch.tensor(0.0))
    log_hourly_obs_scale_scale = pyro.param(
        "log_hourly_obs_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "hourly_obs_scale",
        dist.LogNormal(log_hourly_obs_scale_loc, log_hourly_obs_scale_scale),
    )

    # offset_scale > 0
    log_offset_scale_loc = pyro.param("log_offset_scale_loc", torch.tensor(0.0))
    log_offset_scale_scale = pyro.param(
        "log_offset_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample(
        "offset_scale", dist.LogNormal(log_offset_scale_loc, log_offset_scale_scale)
    )

    # -----------------------------
    # 2) Hourly offset vector
    # -----------------------------
    H = hourly_return.shape[-1]
    hourly_offset_loc = pyro.param("hourly_offset_loc", torch.zeros(H))
    hourly_offset_scale = pyro.param(
        "hourly_offset_scale", 0.1 * torch.ones(H), constraint=constraints.positive
    )
    pyro.sample("hourly_offset", dist.Normal(hourly_offset_loc, hourly_offset_scale))

    # -----------------------------------------------------------------------
    # 3) Correlated weekly latent factors using a MultivariateNormal
    # -----------------------------------------------------------------------
    # We'll have a single random variable "weekly_latent_all" of dimension W.
    # This allows the guide to capture correlations in the approximate posterior.
    # Then, for each weekly_latent_{w_idx} in the model, we use a Delta distribution
    # pinned to all_weeks[w_idx].
    #
    #  weekly_latent_loc: mean vector of length W
    #  weekly_latent_raw_tril: unconstrained lower-triangular part of the covariance
    #                          -> final scale_tril via torch.tril
    #
    #  In the model, each site is weekly_latent_{w_idx}.
    #  In this guide, we "split" the single sample via Delta distributions.
    # -----------------------------------------------------------------------

    weekly_latent_loc = pyro.param("weekly_latent_loc", torch.zeros(W))
    weekly_latent_raw_tril = pyro.param(
        "weekly_latent_raw_tril",
        0.05 * torch.eye(W),  # a small initial scale
        constraint=constraints.lower_cholesky,
    )

    # Build the covariance factor for the correlated distribution
    weekly_latent_scale_tril = torch.tril(weekly_latent_raw_tril)

    # Sample the entire weekly latent vector
    all_weeks = pyro.sample(
        "weekly_latent_all",
        dist.MultivariateNormal(weekly_latent_loc, scale_tril=weekly_latent_scale_tril),
    )

    # For each week in the model, bind weekly_latent_{w_idx} to the corresponding index
    # in our correlated sample using a Delta distribution. This effectively means:
    # weekly_latent_{0} ~ Delta(all_weeks[0]),
    # weekly_latent_{1} ~ Delta(all_weeks[1]), etc.
    #
    # The model sees weekly_latent_{w_idx} as a separate site,
    # but it is actually one element from the single correlated draw all_weeks.
    #
    for w_idx in pyro.markov(range(W)):
        pyro.sample(f"weekly_latent_{w_idx}", dist.Delta(all_weeks[w_idx]))


if __name__ == "__main__":
    daily_return, hourly_return = generate_and_transform_forex_data()

    pyro.render_model(
        model,
        model_args=(daily_return, hourly_return),
        filename="model_dbn.jpg",
        render_distributions=True,
        # render_deterministic=True,
    )

    pyro.render_model(
        guide,
        model_args=(daily_return, hourly_return),
        filename="guide_dbn.jpg",
        render_distributions=True,
        # render_deterministic=True,
    )
