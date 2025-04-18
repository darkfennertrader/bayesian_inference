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
    daily_return,
    hourly_return,
    W=2,
    trading_days=1,
    prior_mean=0.0,
    prior_std=1.0,
):
    """
    A Pyro model capturing a weekly latent factor that drives daily and hourly returns.

    Overview:
      1) We assume a latent state (weekly_latent_w) each week w.
      2) This latent state evolves over time with an AR(1) process:
         weekly_latent_w ~ Normal(phi * weekly_latent_{w-1}, weekly_sigma).
      3) Observations (daily and hourly returns) are modeled via StudentT distributions:
         daily:  StudentT(df_daily, loc=weekly_latent_w, scale=daily_obs_scale)
         hourly: StudentT(df_hourly, loc=weekly_latent_w + hourly_offset, scale=hourly_obs_scale)
         The StudentT distribution helps model heavier tails often seen in financial returns.
      4) An hourly_offset term captures intraday variation around the weekly latent factor.

    Args:
        daily_return (torch.Tensor):
            Observed daily returns, shaped [W, trading_days].
            Each entry daily_return[w, d] is the daily return for day d of week w.
        hourly_return (torch.Tensor):
            Observed hourly returns, shaped [W, trading_days, number_of_hours].
            Each entry hourly_return[w, d, h] is the hourly return for hour h of day d of week w.
        W (int):
            Number of weeks to model.
        trading_days (int):
            Number of trading days per week. Often 5 for Monday-Friday.
        prior_mean (float):
            Prior mean for the first week's latent factor.
        prior_std (float):
            Prior standard deviation for the first week's latent factor.

    Random Variables:
        weekly_sigma: Std for innovations in the AR(1) transition.
        df_daily, df_hourly: Degrees of freedom for the StudentT distributions
                             (daily and hourly, respectively).
        daily_obs_scale, hourly_obs_scale: Scale parameters for daily/hourly StudentT.
        offset_scale: Scale for the Normal distribution from which hourly_offset is sampled.
        hourly_offset: Intraday offset vector, shape [number_of_hours].
        phi: AR(1) coefficient controlling the memory in latent factors.
        weekly_latent_w: Latent factor for week w.
    """

    # 1) Weekly transition noise standard deviation for AR(1)
    weekly_sigma = pyro.sample("weekly_sigma", dist.Exponential(torch.tensor(1.0)))

    # 2) StudentT distribution parameters for daily and hourly returns
    #    (degrees of freedom and scales).
    df_daily = pyro.sample("df_daily", dist.Exponential(torch.tensor(0.5)))
    df_hourly = pyro.sample("df_hourly", dist.Exponential(torch.tensor(0.5)))
    daily_obs_scale = pyro.sample("daily_obs_scale", dist.Exponential(torch.tensor(1.0)))
    hourly_obs_scale = pyro.sample("hourly_obs_scale", dist.Exponential(torch.tensor(1.0)))

    # 3) Hourly offsets capture intraday variation around the weekly latent factor.
    H = hourly_return.shape[-1]
    offset_scale = pyro.sample("offset_scale", dist.Exponential(torch.tensor(1.0)))
    hourly_offset = pyro.sample(
        "hourly_offset",
        dist.Normal(torch.zeros(H), offset_scale * torch.ones(H)),
    )

    # AR(1) coefficient
    phi = pyro.sample("phi", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))

    # 4) Keep track of previous week's latent factor. The very first one has a fixed prior.
    weekly_latent_prev = torch.tensor(prior_mean)

    # 5) Loop over weeks with AR(1) dynamics:
    # weekly_latent_w ~ Normal(phi*weekly_latent_{w-1}, weekly_sigma).
    for w_idx in pyro.markov(range(W)):  # pyro.markov helps Pyro handle sequential dependencies

        # For the first week, use the prior mean and sd directly
        if w_idx == 0:
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev, prior_std),
            )
        else:
            # AR(1) update for the weekly latent factor
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(phi * weekly_latent_prev, weekly_sigma),
            )

        # 6) Condition daily and hourly observations on the weekly latent factor
        for d_idx in range(trading_days):
            # Daily observation
            pyro.sample(
                f"daily_obs_{w_idx}_{d_idx}",
                dist.StudentT(df_daily, weekly_latent_w, daily_obs_scale),
                obs=daily_return[w_idx, d_idx],
            )

            # Hourly observations
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

        # Update "previous" latent factor for the next week
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
        Variational Guide for the AR(1) weekly latent factor model.

        Overview:
          - This guide approximates the posterior distribution of all latent variables
            using a combination of (a) mean-field distributions for global parameters,
            and (b) a MultivariateNormal distribution for the entire sequence of
            weekly_latent_0, ..., weekly_latent_(W-1). That approach allows
            correlations among the weekly latent factors to be learned by SVI.
          - We place independent LogNormal or Normal distributions on the other
            parameters (df_daily, df_hourly, daily_obs_scale, etc.).

        Args:
            daily_return (torch.Tensor): Same shape as in the model, [W, trading_days].
            hourly_return (torch.Tensor): Same shape as in the model, [W, trading_days, number_of_hours]
            W (int): Number of weeks.
            trading_days (int): Number of trading days per week.
            prior_mean (float): Prior mean for the first week's latent factor (unused here directly).
            prior_std (float): Prior std for the first weekfrom pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam's latent factor (unused here directly).

        Random Variables (elements of the variational distribution):
            weekly_sigma:   LogNormal(mean=log_weekly_sigma_loc, std=log_weekly_sigma_scale).
            df_daily:       LogNormal for daily degrees of freedom.
            df_hourly:      LogNormal for hourly degrees of freedom.
            daily_obs_scale, hourly_obs_scale, offset_scale: LogNormal distributions.
            hourly_offset:  Normal with learned location anfrom pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adamd scale for intraday offsets.
            phi:            AR(1) coefficient as a Normal with learned location and scale.
            weekly_latent_all: MultivariateNormal capturing correlation in the
                               weekly latent factors. Then each weekly_latent_w is
                               given by Delta(all_weeks[w_idx]) to tie it to this
                               single distribution sample.
    """

    # -----------------------------
    # 1) Global AR(1) transition scale (weekly_sigma)
    # -----------------------------
    log_weekly_sigma_loc = pyro.param("log_weekly_sigma_loc", torch.tensor(0.0))
    log_weekly_sigma_scale = pyro.param(
        "log_weekly_sigma_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("weekly_sigma", dist.LogNormal(log_weekly_sigma_loc, log_weekly_sigma_scale))

    # -----------------------------
    # 2) Degrees of freedom for the StudentT distributions
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
    # 3) Scales for the StudentT distributions (daily, hourly)
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
    # 4) Offset scale for intraday variations and the offsets themselves
    # -----------------------------
    log_offset_scale_loc = pyro.param("log_offset_scale_loc", torch.tensor(0.0))
    log_offset_scale_scale = pyro.param(
        "log_offset_scale_scale", torch.tensor(0.1), constraint=constraints.positive
    )
    pyro.sample("offset_scale", dist.LogNormal(log_offset_scale_loc, log_offset_scale_scale))

    # Hourly offset vector
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

    # --------------------------------------------------------------
    # 6) Correlated weekly latent factors using a MultivariateNormal
    # --------------------------------------------------------------
    # We place the entire vector (weekly_latent_0, ..., weekly_latent_(W-1))
    # into one MultivariateNormal distribution so that SVI can learn
    # correlations between weeks.
    weekly_latent_loc = pyro.param("weekly_latent_loc", torch.zeros(W))
    weekly_latent_raw_tril = pyro.param(
        "weekly_latent_raw_tril",
        0.05 * torch.eye(W),
        constraint=constraints.lower_cholesky,
    )
    weekly_latent_scale_tril = torch.tril(weekly_latent_raw_tril)

    # Sample the vector for all weeks
    all_weeks = pyro.sample(
        "weekly_latent_all",
        dist.MultivariateNormal(weekly_latent_loc, scale_tril=weekly_latent_scale_tril),
    )

    # Because the model code references weekly_latent_w individually,
    # we tie each weekly_latent_{w_idx} to the relevant element of all_weeks
    # via a Delta distribution.
    # This keeps the shapes consistent while capturing correlations.
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
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
