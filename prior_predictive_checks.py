import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pyro
from pyro.infer import Predictive
import dbn_ar1_garch11_Markov_switching_regime as bn


def extract_obs(samples, obs_name_prefix):
    """
    Collects (and flattens) all samples that begin with a specified prefix,
    such as 'daily_obs_' or 'hourly_obs_'.

    Args:
        samples (dict): Dictionary returned by pyro.infer.Predictive.
        obs_name_prefix (str): Prefix for observation keys, e.g. 'daily_obs_'.

    Returns:
        np.ndarray of all the extracted and flattened observations.
    """

    obs_keys = [k for k in samples.keys() if k.startswith(obs_name_prefix)]
    all_obs = []
    for k in obs_keys:
        obs_tensor = samples[k].detach().cpu().numpy()
        all_obs.append(obs_tensor.flatten())

    if len(all_obs) == 0:
        return np.array([])

    return np.concatenate(all_obs, axis=0)


def plot_qq_studentT(data_array, df, loc, scale, ax=None, title_suffix=""):
    """
    Creates a QQ plot comparing data_array to a StudentT(df, loc=loc, scale=scale).

    data_array: 1D array of observations (floats).
    df: degrees of freedom for the T distribution.
    loc: location (mean) of the T distribution.
    scale: scale of the T distribution.
    ax: optional matplotlib Axes. If None, a new figure is created.
    title_suffix: Optional string to append to the plot title.
    """
    # If no axis is provided, create a new figure and axis
    created_new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_new_fig = True

    sorted_data = np.sort(data_array)
    n = len(sorted_data)
    probs = (np.arange(n) + 0.5) / n

    # Compute theoretical quantiles from the chosen T distribution
    t_theoretical = stats.t.ppf(probs, df, loc=loc, scale=scale)

    ax.scatter(t_theoretical, sorted_data, alpha=0.5, label="Empirical vs. Theoretical")

    # Fit a least-squares line for a visual check (how close to y = x?)
    slope, intercept = np.polyfit(t_theoretical, sorted_data, 1)
    x_vals = np.linspace(t_theoretical.min(), t_theoretical.max(), 100)
    ax.plot(x_vals, slope * x_vals + intercept, "r", label="Least-squares fit")

    ax.set_title(
        f"QQ Plot vs. StudentT(df={df:.2f}, loc={loc:.2f}, scale={scale:.2f}) {title_suffix}"
    )
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Data")
    ax.legend()

    # Only save/show if we created a new figure
    if created_new_fig:
        plt.tight_layout()
        plt.show()


def plot_prior_distributions_grid(samples, param_list):
    """
    Displays histograms of parameter samples in a grid with 3 columns.
    Also overlays a text box (mean, std, min, max) in the upper-right
    corner of each subplot.

    Additionally, this version plots two extra subplots (if daily returns
    are found in the samples):
    1) A histogram + KDE of the daily returns (prior).
    2) A QQ-plot comparing the daily returns to a representative Student T distribution.
    """

    n_params = len(param_list)
    daily_returns_prior = extract_obs(samples, obs_name_prefix="daily_obs_")
    has_daily_returns = daily_returns_prior.size > 0 and "df_daily_offset" in samples

    n_plots = n_params
    if has_daily_returns:
        n_plots += 2  # One for the histogram, one for the QQ plot

    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    # 2) Plot the parameter distributions
    for i, param in enumerate(param_list):
        param_values = samples[param].detach().cpu().numpy().flatten()
        p_mean = param_values.mean()
        p_std = param_values.std()
        p_min = param_values.min()
        p_max = param_values.max()

        sns.histplot(param_values, kde=True, stat="density", ax=axes[i])
        axes[i].set_title(f"Prior Distribution: {param}", fontsize=10)

        summary_text = (
            f"Mean: {p_mean:.4f}\n"
            f"Std:  {p_std:.4f}\n"
            f"Min:  {p_min:.4f}\n"
            f"Max:  {p_max:.4f}"
        )
        axes[i].text(
            0.95,
            0.95,
            summary_text,
            transform=axes[i].transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.6),
        )

    # 3) If daily observations exist, create two extra subplots:
    #    - One for daily returns histogram
    #    - One for the QQ plot
    current_axis_index = n_params
    if has_daily_returns:
        # 3a) Daily histogram + KDE
        daily_hist_ax = axes[current_axis_index]
        sns.histplot(daily_returns_prior, kde=True, bins=50, ax=daily_hist_ax)
        daily_hist_ax.set_title("Daily Returns (Prior) - Histogram + KDE", fontsize=10)

        # Compute mean, std, skewness, and kurtosis
        ret_mean = daily_returns_prior.mean()
        ret_std = daily_returns_prior.std()
        ret_skew = stats.skew(daily_returns_prior)
        ret_kurt = stats.kurtosis(daily_returns_prior)

        daily_stats_text = (
            f"Mean: {ret_mean:.4f}\n"
            f"Std:  {ret_std:.4f}\n"
            f"Skew: {ret_skew:.4f}\n"
            f"Kurt: {ret_kurt:.4f}"
        )
        daily_hist_ax.text(
            0.95,
            0.95,
            daily_stats_text,
            transform=daily_hist_ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.6),
        )

        current_axis_index += 1

        # 3b) QQ Plot
        qq_ax = axes[current_axis_index]
        df_samples = samples["df_daily_offset"].detach().cpu().numpy().flatten()
        df_daily_samples = 2.0 + df_samples
        df_rep = np.median(df_daily_samples)  # or mean, if preferred

        loc_rep = daily_returns_prior.mean()
        scale_rep = daily_returns_prior.std()

        plot_qq_studentT(
            daily_returns_prior,
            df=df_rep,
            loc=loc_rep,
            scale=scale_rep,
            ax=qq_ax,
            title_suffix="(Daily Obs)",
        )
        qq_ax.set_title("QQ Plot vs. StudentT - (Daily Returns Prior)", fontsize=10)

    # 4) Disable any leftover axes
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=3.0)
    fig.savefig("predictive_priority_checks.png", dpi=300)
    plt.show()


def prior_predictive_check(model, W=2, trading_days=1, H=24, num_samples=100):
    """
    Generate samples from the model using only priors (no real data),
    then return a dict containing prior-sampled parameters and observations.
    """
    predictive = Predictive(model=model, num_samples=num_samples)
    samples = predictive(
        daily_return=None,
        hourly_return=None,
        W=W,
        trading_days=trading_days,
        prior_mean=0.0,
        prior_std=1.0,
        prior_predictive_checks=True,
    )
    return samples


# Set the seed for reproducibility
pyro.set_rng_seed(42)
samples = prior_predictive_check(bn.model)
param_list = [
    "garch_omega",
    "alpha_beta_sum",
    "alpha_frac",
    "df_daily_offset",
    "df_hourly_offset",
    "daily_obs_scale",
    "hourly_obs_scale",
    "offset_scale",
    "phi",
    "regime_means",
    "regime_volatility",
]
plot_prior_distributions_grid(samples, param_list)
