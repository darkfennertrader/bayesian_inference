import math
import pandas as pd
import torch
from arch import arch_model


def fit_garch(filepath, scale_returns=100, return_col="eur-usd", date_col="Datetime"):
    """
    Fit a GARCH(1,1) model on historical return data and return the estimated parameters.

    Parameters:
    filepath     : str
                    Path to the CSV file containing the data. The CSV should have at least
                    two columns: one for datetime information and one for the asset's return.
    scale_returns: float, default 100
                    Factor to scale the returns. Since you plan to use returns*100 in SVI,
                    scaling here ensures consistency.
    return_col   : str, default 'eur-usd'
                    Name of the column that already contains the computed returns in the form
                    (r(t)-r(t-1))/r(t-1).
    date_col     : str, default 'Datetime'
                    Name of the column with date/time information.

    Returns:
    params       : dict
                    A dictionary with the estimated parameters (including the constant mean 'mu',
                    and the GARCH parameters 'omega', 'alpha[1]', and 'beta[1]').
    """
    # Step 1: Read the data, parsing the date column and setting it as the index.
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # Step 2: Since the returns in return_col are already computed as (r(t)-r(t-1))/r(t-1),
    # we simply scale them by scale_returns (typically 100) to be consistent with later SVI inference.
    df["return_scaled"] = df[return_col] * scale_returns

    # Step 3: Define and fit the GARCH(1,1) model.
    # We use a constant mean and assume normally distributed errors.
    am = arch_model(df["return_scaled"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(update_freq=0, disp="off")

    # Optionally print the summary for inspection
    print(res.summary())

    # Step 4: Extract and return the estimated parameters.
    # Parameter names typically include 'mu' (the constant mean), 'omega' (baseline volatility),
    # 'alpha[1]' (coefficient on lagged squared residuals), and 'beta[1]' (coefficient on lagged volatility)
    params = res.params.to_dict()
    return params


def transform_garch_parameters(fitted_params):
    """
    Transform the GARCH(1,1) parameters from a historical arch model fit so that they can be used
    as initial values in the pyro model.

    Parameters:
        fitted_params : dict
            A dictionary containing the fitted GARCH parameters with keys
            'omega', 'alpha[1]' and 'beta[1]'.

    Returns:
        dict with keys:
            'garch_omega'       : the base variance parameter,
            'alpha_beta_sum'    : the sum of the alpha and beta parameters,
            'alpha_frac'        : the proportion for splitting the sum between alpha and beta,
            'garch_sigma_init'  : the initial volatility computed using unconditional variance.
    """

    # Extract fitted parameters.
    omega = fitted_params["omega"]
    alpha1 = fitted_params["alpha[1]"]
    beta1 = fitted_params["beta[1]"]

    # Compute the sum. This is used to ensure the stationarity condition (alpha + beta < 1).
    alpha_beta_sum = alpha1 + beta1
    if alpha_beta_sum <= 0:
        raise ValueError("The sum of alpha[1] and beta[1] must be positive.")
    if alpha_beta_sum >= 1:
        raise ValueError(
            "The stationarity condition requires alpha[1] + beta[1] to be less than 1."
        )

    # Determine the fraction for alpha.
    alpha_frac = alpha1 / alpha_beta_sum

    # Compute the unconditional (initial) volatility through the GARCH(1,1) formula:
    # sigma^2 = omega / (1 - (alpha + beta))
    sigma_prev = math.sqrt(omega / (1 - alpha_beta_sum))

    # Depending on your usage in the pyro model you might want to convert these to torch tensors.
    return {
        "garch_omega": torch.tensor(omega, dtype=torch.float),
        "alpha_beta_sum": torch.tensor(alpha_beta_sum, dtype=torch.float),
        "alpha_frac": torch.tensor(alpha_frac, dtype=torch.float),
        "garch_sigma_init": torch.tensor(sigma_prev, dtype=torch.float),
    }


if __name__ == "__main__":
    filepath = "data/processed_returns.csv"
    fit_garch(filepath, scale_returns=100, return_col="eur-usd", date_col="Datetime")
