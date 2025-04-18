from typing import Optional
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex
from torch.distributions import constraints


# Step 1: Identify Parameters Clearly
# Your model has the following parameters (for a single asset):

# Global Parameters (common across regimes):
# GARCH(1,1) parameters:

# omega (1 parameter)
# alpha (1 parameter)
# beta (1 parameter)
# sigma_init (1 parameter)
# Student-T parameters:

# df_daily (1 parameter)
# df_hourly (1 parameter)
# daily_obs_scale (1 parameter)
# hourly_obs_scale (1 parameter)
# Hourly offsets:

# offset_scale (1 parameter)
# hourly_offset (H parameters, where H = number of hourly intervals per trading day)
# AR(1) coefficient:

# phi (1 parameter)
# Markov Switching Model (3 regimes):
# Initial regime probabilities:

# regime_init_probs (3 parameters, but sum-to-one constraint means effectively 2 free parameters)
# Transition matrix (3x3):

# regime_transition_probs (3 regimes × 3 probabilities each = 9 parameters, but each row sums to one, so effectively 3 × (3-1) = 6 free parameters)
# Regime-specific parameters:

# regime_means (3 parameters, one per regime)
# regime_volatility (3 parameters, one per regime)
# Weekly latent variables:
# weekly_latent (latent factor per week): W parameters, where W is the number of weeks in your dataset.
# Step 2: Count Parameters Explicitly (for a single asset):
# Let's assume hourly data for a typical trading day (e.g., 24 hourly intervals per day):

# GARCH parameters: 4
# Student-T parameters: 4
# Hourly offsets: 1 (offset_scale) + 24 (hourly_offset) = 25
# AR(1) coefficient: 1
# MSM parameters:
# Initial regime probabilities: 2 (effective)
# Transition probabilities: 6 (effective)
# Regime means: 3
# Regime volatilities: 3
# Total fixed parameters (excluding weekly latent variables):

# 4 (GARCH) + 4 (Student-T) + 25 (hourly offsets) + 1 (AR) + 2 (initial regime probs) + 6 (transition probs) + 3 (regime means) + 3 (regime volatilities) = 48 parameters

# Step 3: Count Latent Variables (weekly latent factors):
# For each week, you have one latent variable (weekly_latent).
# For 5 years of data, assuming approximately 52 weeks per year:
# W = 5 years × 52 weeks/year = 260 latent variables.
# Thus, total latent variables for 5 years = 260.

# Step 4: Total Parameters and Latent Variables:
# Fixed parameters: 48
# Latent variables (weekly): 260
# Total parameters and latent variables: 48 + 260 = 308
# Step 5: Data Availability and Sufficiency:
# You mentioned:

# 1 year of data ≈ 12,000 hourly data points per asset (or 6,000 after removing NaNs).
# For 5 years, you have approximately 30,000 hourly data points (after NaN removal).
# Additionally, you have daily returns (approximately 260 trading days per year × 5 years ≈ 1,300 daily data points).
# Thus, total data points for inference (5 years):

# Hourly data: ~30,000 points
# Daily data: ~1,300 points
# Total ≈ 31,300 data points
# You have about 31,300 data points to estimate 308 parameters and latent variables. This gives a data-to-parameter ratio of approximately:

# 31,300 / 308 ≈ 101.6 data points per parameter/latent variable.

# This ratio (~100 data points per parameter) is generally considered sufficient for stable inference, especially when using Bayesian inference with informative priors and hierarchical structures.

# Step 6: Weekly Rolling Update (2-year window):
# After the initial 5-year training, you plan to update weekly using a rolling 2-year window:

# 2 years ≈ 104 weeks → 104 latent variables
# Fixed parameters remain the same (48 parameters)
# Total parameters and latent variables for rolling updates: 48 + 104 = 152
# Data points for rolling 2-year window:

# Hourly data: 2 years × 6,000 points/year ≈ 12,000 points
# Daily data: 2 years × 260 days/year ≈ 520 points
# Total ≈ 12,520 data points
# Data-to-parameter ratio for rolling updates:

# 12,520 / 152 ≈ 82.4 data points per parameter/latent variable.

# This ratio (~82 data points per parameter) is still acceptable, though slightly lower. It remains sufficient for stable inference, especially given that you have already trained the model initially on 5 years of data, providing good priors and initialization.

# Step 7: Strengths and Weaknesses of this Setup:
# Strengths:
# Good data-to-parameter ratio (~100 initially, ~82 rolling), sufficient for stable inference.
# Bayesian inference (SVI) with informative priors helps stabilize parameter estimation.
# Rolling updates allow the model to adapt dynamically to changing market conditions.
# Weaknesses:
# The complexity of the model (308 parameters initially, 152 rolling) still requires careful tuning and validation.
# Potential identifiability issues remain, especially for regime parameters and hourly offsets.
# Computational cost may be significant, requiring careful optimization and possibly GPU acceleration.
# Recommendations:
# Start with the initial 5-year training to obtain stable parameter estimates.
# Carefully monitor convergence diagnostics (ELBO, parameter stability, posterior predictive checks).
# Consider simplifying the model initially (e.g., fewer regimes or fewer hourly offsets) and gradually increase complexity.
# Use informative priors to stabilize inference, especially for regime parameters and GARCH parameters.
# Regularly validate the model's predictive performance and regime identification against known market events.
# Conclusion:
# Your proposed setup (5 years initial training, weekly rolling 2-year updates) provides sufficient data for stable inference given your model complexity. The data-to-parameter ratio (~100 initially, ~82 rolling) is acceptable. However, careful implementation, validation, and possibly incremental complexity are recommended to ensure robust and reliable inference.


# General Guidelines for Data-to-Parameter Ratios:
# Less than 5 data points per parameter:
# Usually insufficient. High risk of overfitting, unstable inference, and poor identifiability.

# 5–10 data points per parameter:
# Marginally acceptable. Requires strong informative priors, careful regularization, and validation.

# 10–20 data points per parameter:
# Generally acceptable. Stable inference achievable with moderate priors and careful tuning.

# 20–50 data points per parameter:
# Good. Typically stable inference, robust parameter estimation, and reliable posterior uncertainty quantification.

# 50–100+ data points per parameter:
# Excellent. Highly stable inference, robust parameter estimation, and reliable uncertainty quantification. Allows more complex models and weaker priors.


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
    daily_return: Optional[torch.Tensor] = None,  # shape: [W, trading_days]
    hourly_return: Optional[torch.Tensor] = None,  # shape: [W, trading_days, H]
    W=2,
    trading_days=1,
    prior_mean=0.0,
    prior_std=1.0,
    prior_predictive_checks=False,
):
    # W is the number of weeks we are modeling.
    # trading_days is how many trading days exist in each week. For a typical scenario you might set trading_days=5, but here it’s set to 1 for simplicity.
    # prior_mean and prior_std define a prior for the very first latent factor.
    # prior_predictive_checks is a flag that can switch between sampling from prior vs. using real observed data.

    # Adapt H to the shape of the hourly_return if provided; otherwise assume 24-hour structure.
    H = (
        hourly_return.shape[-1] if hourly_return is not None else 24
    )  # default hourly dimension if None

    # -----------------------------
    # 1) GARCH(1,1) parameters (common across regimes)
    # -----------------------------
    # garch_omega is the base variance parameter in GARCH.
    garch_omega = pyro.sample("garch_omega", dist.Exponential(10.0))

    # alpha_beta_sum ensures alpha + beta < 1; we sample a Beta(9,1), then split that into alpha/beta.
    alpha_beta_sum = pyro.sample("alpha_beta_sum", dist.Beta(9.0, 1.0))
    alpha_frac = pyro.sample("alpha_frac", dist.Beta(2.0, 2.0))

    garch_alpha = alpha_beta_sum * alpha_frac
    garch_beta = alpha_beta_sum * (1.0 - alpha_frac)

    # sigma_prev is the initial volatility for the GARCH(1,1) recursion.
    sigma_prev = pyro.sample("garch_sigma_init", dist.Exponential(10.0))

    # -----------------------------
    # 2) StudentT parameters (common across regimes)
    # -----------------------------
    df_daily_offset = pyro.sample("df_daily_offset", dist.Gamma(2.0, 0.5))
    df_daily = 2.0 + df_daily_offset

    df_hourly_offset = pyro.sample("df_hourly_offset", dist.Gamma(2.0, 0.5))
    df_hourly = 2.0 + df_hourly_offset
    # These define how noisy the daily vs. hourly observations are, under a Student T.
    daily_obs_scale = pyro.sample("daily_obs_scale", dist.Exponential(1.0))
    hourly_obs_scale = pyro.sample("hourly_obs_scale", dist.Exponential(1.0))

    # -----------------------------
    # 3) Hourly offsets (common across regimes)
    # -----------------------------
    # Each hour has its own offset from the weekly latent factor.
    offset_scale = pyro.sample("offset_scale", dist.Exponential(1.0))
    hourly_offset = pyro.sample(
        "hourly_offset",
        dist.Normal(torch.zeros(H), offset_scale * torch.ones(H)),
    )

    # -----------------------------
    # 4) AR(1) coefficient (common across regimes)
    # -----------------------------
    # This is the AR(1) coefficient controlling how weekly latent factors depend on prior weeks.
    phi = pyro.sample("phi", dist.Normal(0.0, 1.0))

    # -----------------------------
    # 5) Markov Switching Model (3 regimes)
    # -----------------------------
    num_regimes = 3

    # The initial probabilities of being in each regime
    regime_init_probs = pyro.sample("regime_init_probs", dist.Dirichlet(torch.ones(num_regimes)))

    # The transition matrix among the 3 regimes (3x3)
    regime_transition_probs = pyro.sample(
        "regime_transition_probs",
        dist.Dirichlet(torch.ones(num_regimes, num_regimes)).to_event(1),
    )

    # Regime-specific means, capturing a shift in the weekly latent factor, Narrower prior on regime_means to reflect +/-1% typical drift when scaled by 100
    regime_means = pyro.sample(
        "regime_means", dist.Normal(torch.zeros(num_regimes), 0.5 * torch.ones(num_regimes))
    )
    # Regime-specific volatility multipliers, capturing how the GARCH volatility might differ.Tighter prior for regime_volatility to avoid extreme volatility draws in x100 space
    regime_volatility = pyro.sample(
        "regime_volatility",
        dist.LogNormal(torch.tensor([-0.5, 0.0, 0.5]), 0.5 * torch.ones(num_regimes)),
    )

    # Enumerate the discrete regime variables efficiently within pyro.markov
    regimes = []
    with pyro.markov():
        regime_prev = pyro.sample(
            "regime_0", dist.Categorical(regime_init_probs), infer={"enumerate": "sequential"}
        )
        regimes.append(regime_prev)

        for w_idx in range(1, W):
            regime_curr = pyro.sample(
                f"regime_{w_idx}",
                dist.Categorical(Vindex(regime_transition_probs)[regime_prev]),
                infer={"enumerate": "sequential"},
            )
            regimes.append(regime_curr)
            regime_prev = regime_curr

    # -----------------------------
    # 6) Weekly latent factors with GARCH update
    # -----------------------------
    weekly_latent_prev = torch.tensor(prior_mean)
    e_prev = torch.tensor(0.0)

    for w_idx in pyro.markov(range(W)):
        current_regime = regimes[w_idx]
        regime_mean = Vindex(regime_means)[current_regime]
        regime_vol = Vindex(regime_volatility)[current_regime]

        if w_idx == 0:
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(weekly_latent_prev + regime_mean, prior_std),
            )
            e_prev = weekly_latent_w - weekly_latent_prev
        else:
            # GARCH update for volatility
            sigma_current_sq = (
                garch_omega + garch_alpha * (e_prev**2) + garch_beta * (sigma_prev**2)
            )
            sigma_current = torch.sqrt(sigma_current_sq) * regime_vol

            # AR(1): weekly_latent_w depends on phi * old latent + a shift from the regime mean
            weekly_latent_w = pyro.sample(
                f"weekly_latent_{w_idx}",
                dist.Normal(phi * weekly_latent_prev + regime_mean, sigma_current),
            )

            e_prev = weekly_latent_w - (phi * weekly_latent_prev + regime_mean)
            sigma_prev = sigma_current

        # Observations for each day in that week
        for d_idx in range(trading_days):
            daily_obs_dist = dist.StudentT(df_daily, weekly_latent_w, daily_obs_scale)
            # If prior_predictive_checks is True, we set obs=None
            # so we just sample from the prior or posterior predictive
            if prior_predictive_checks or (daily_return is None):
                curr_daily_obs = None
            else:
                curr_daily_obs = daily_return[w_idx, d_idx]
                if torch.isnan(curr_daily_obs):
                    curr_daily_obs = None

            pyro.sample(
                f"daily_obs_{w_idx}_{d_idx}",
                daily_obs_dist,
                obs=curr_daily_obs,
            )

            # Observations for each hour in that day
            for h_idx in range(H):
                hourly_obs_dist = dist.StudentT(
                    df_hourly,
                    weekly_latent_w + hourly_offset[h_idx],
                    hourly_obs_scale,
                )

                if prior_predictive_checks or (hourly_return is None):
                    curr_hourly_obs = None
                else:
                    curr_hourly_obs = hourly_return[w_idx, d_idx, h_idx]
                    if torch.isnan(curr_hourly_obs):
                        curr_hourly_obs = None

                pyro.sample(
                    f"hourly_obs_{w_idx}_{d_idx}_{h_idx}",
                    hourly_obs_dist,
                    obs=curr_hourly_obs,
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
    H = hourly_return.shape[-1]
    num_regimes = 3

    # GARCH parameters (unchanged)
    pyro.sample(
        "garch_omega", dist.LogNormal(pyro.param("garch_omega_loc", torch.tensor(0.1)), 0.1)
    )
    pyro.sample(
        "alpha_beta_sum",
        dist.Beta(
            pyro.param("ab_sum_conc1", torch.tensor(9.0)),
            pyro.param("ab_sum_conc0", torch.tensor(1.0)),
        ),
    )
    pyro.sample(
        "alpha_frac",
        dist.Beta(
            pyro.param("alpha_frac_conc1", torch.tensor(2.0)),
            pyro.param("alpha_frac_conc0", torch.tensor(2.0)),
        ),
    )
    pyro.sample(
        "garch_sigma_init", dist.LogNormal(pyro.param("sigma_init_loc", torch.tensor(0.1)), 0.1)
    )

    # StudentT parameters
    df_daily_offset_shape = pyro.param(
        "df_daily_offset_shape", torch.tensor(2.0), constraint=constraints.positive
    )
    df_daily_offset_rate = pyro.param(
        "df_daily_offset_rate", torch.tensor(0.5), constraint=constraints.positive
    )
    pyro.sample("df_daily_offset", dist.Gamma(df_daily_offset_shape, df_daily_offset_rate))

    # CHANGED: same for "df_hourly_offset"
    df_hourly_offset_shape = pyro.param(
        "df_hourly_offset_shape", torch.tensor(2.0), constraint=constraints.positive
    )
    df_hourly_offset_rate = pyro.param(
        "df_hourly_offset_rate", torch.tensor(0.5), constraint=constraints.positive
    )
    pyro.sample("df_hourly_offset", dist.Gamma(df_hourly_offset_shape, df_hourly_offset_rate))

    pyro.sample(
        "daily_obs_scale", dist.LogNormal(pyro.param("daily_obs_scale_loc", torch.tensor(0.1)), 0.1)
    )
    pyro.sample(
        "hourly_obs_scale",
        dist.LogNormal(pyro.param("hourly_obs_scale_loc", torch.tensor(0.1)), 0.1),
    )

    # Hourly offsets
    pyro.sample(
        "offset_scale", dist.LogNormal(pyro.param("offset_scale_loc", torch.tensor(0.1)), 0.1)
    )
    pyro.sample(
        "hourly_offset",
        dist.Normal(
            pyro.param("hourly_offset_loc", torch.zeros(H)),
            pyro.param("hourly_offset_scale", 0.1 * torch.ones(H)),
        ),
    )

    # AR(1) coefficient
    pyro.sample(
        "phi",
        dist.Normal(
            pyro.param("phi_loc", torch.tensor(0.0)), pyro.param("phi_scale", torch.tensor(0.1))
        ),
    )
    pyro.sample(
        "regime_means",
        dist.Normal(
            pyro.param("regime_means_loc", torch.zeros(num_regimes)),
            pyro.param(
                "regime_means_scale", torch.ones(num_regimes), constraint=constraints.positive
            ),
        ),
    )
    pyro.sample(
        "regime_volatility",
        dist.LogNormal(
            pyro.param("regime_volatility_loc", torch.zeros(num_regimes)),
            pyro.param(
                "regime_volatility_scale", torch.ones(num_regimes), constraint=constraints.positive
            ),
        ),
    )

    # Weekly latent factors, each with its own variational loc and scale
    weekly_latent_loc = pyro.param("weekly_latent_loc", torch.zeros(W))
    weekly_latent_scale = pyro.param(
        "weekly_latent_scale", 0.1 * torch.ones(W), constraint=constraints.positive
    )
    for w_idx in range(W):
        pyro.sample(
            f"weekly_latent_{w_idx}",
            dist.Normal(weekly_latent_loc[w_idx], weekly_latent_scale[w_idx]),
        )

    # ------------------------
    # Discrete states via guide-side SVI enumeration
    # ------------------------

    regime_probs0 = pyro.param(
        "regime_q_0",
        torch.ones(num_regimes),
        constraint=constraints.simplex,
    )
    regime_prev = pyro.sample(
        "regime_0", dist.Categorical(regime_probs0), infer={"enumerate": "sequential"}
    )

    with pyro.markov():
        for w_idx in range(1, W):
            regime_probs_t = pyro.param(
                f"regime_q_{w_idx}",
                torch.ones(num_regimes, num_regimes),
                constraint=constraints.simplex,
            )
            regime_curr = pyro.sample(
                f"regime_{w_idx}",
                dist.Categorical(regime_probs_t[regime_prev]),
                infer={"enumerate": "sequential"},
            )
            regime_prev = regime_curr


if __name__ == "__main__":
    daily_return, hourly_return = generate_and_transform_forex_data()

    # print(daily_return)
    # print(type(daily_return))

    # pyro.render_model(
    #     model,
    #     model_args=(daily_return, hourly_return),
    #     filename="model_dbn_ar1_garch11_MSM_false.jpg",
    #     render_params=True,
    #     render_distributions=True,
    #     # render_deterministic=True,
    # )

    # pyro.render_model(
    #     guide,
    #     model_args=(daily_return, hourly_return),
    #     filename="guide_dbn_ar1_garch11_MSM.jpg",
    #     render_params=True,
    #     render_distributions=True,
    #     # render_deterministic=True,
    # )
