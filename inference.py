import math
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO

# pylint: disable=no-name-in-module
from pyro.optim import Adam  #  type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pre_processing import load_and_prepare_returns
import dbn_ar1_garch11_Markov_switching_regime as bn


# 1 - Data Preparation and pre-processing
# hourly_returns, daily_returns = load_and_prepare_returns("data/processed_returns.csv")
# hourly_returns *= 100  # scale returns for numerical stability
# print(hourly_returns[:, 0])

# Step 2: Model Sanity Check and Initialization
# Prior Predictive Checks: Before inference, perform prior predictive sampling to ensure your priors are reasonable and produce realistic simulated data.
# Initialization Optional): Initialize parameters carefully. Poor initialization can lead to slow convergence or divergence. Consider initializing parameters close to empirical estimates (e.g., GARCH parameters from historical volatility estimates).

# Step 3: Inference Setup (SVI)
# Optimizer Choice: Use Adam optimizer with a small learning rate (e.g., 0.005 or 0.001) initially. Adjust learning rate dynamically if needed.
# Guide Design: Your guide (variational distribution) should closely match the complexity of your model. Ensure all parameters have appropriate constraints and initializations.
# Elbo Loss Monitoring: Track the Evidence Lower Bound (ELBO) loss during inference. ELBO should decrease smoothly. If it fluctuates or diverges, revisit initialization, priors, or guide structure.


def run_svi_inference(model, guide, daily_return, hourly_return, num_steps=1000, lr=0.01):
    """
    Runs SVI with the provided daily and hourly returns.

    daily_return: a tensor of shape (W, trading_days)
    hourly_return: a tensor of shape (W, trading_days, H)
    num_steps: the number of SVI iterations
    lr: learning rate for the Adam optimizer

    Returns a tuple (losses, parameter_store) with the list of ELBO losses and the optimized pyro parameters.
    """
    # Clear any existing parameters
    pyro.clear_param_store()

    # Determine the number of weeks (W) and trading days from the data shapes.
    W = daily_return.shape[0]
    trading_days = daily_return.shape[1]

    # Set up the optimizer.
    optimizer = Adam({"lr": lr})

    # Use TraceEnum_ELBO to correctly marginalize enumerated discrete latent variables.
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses = []
    for step in range(num_steps):
        # Note: we are passing our data and the proper W and trading_days to the model/guide.
        loss = svi.step(
            daily_return=daily_return, hourly_return=hourly_return, W=W, trading_days=trading_days
        )
        losses.append(loss)

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

    print(f"Final loss after {num_steps} iterations: {losses[-1]:.4f}")
    return losses, pyro.get_param_store()


if __name__ == "__main__":
    data = torch.load("data/assets.pt", weights_only=False)
    # check shapes
    print(data["assets"]["eur-usd"]["hourly_return"].shape)
    print(data["assets"]["eur-usd"]["daily_return"].shape)

    daily_return = data["assets"]["eur-usd"]["daily_return"]
    # print(daily_return.shape[0])
    hourly_return = data["assets"]["eur-usd"]["hourly_return"]

    print(f"daily_return shape: {daily_return.shape}")
    print(f"hourly_return shape: {hourly_return.shape}")

    # Run the SVI inference
    losses, param_store = run_svi_inference(
        bn.model, bn.guide, daily_return, hourly_return, num_steps=1000, lr=0.01
    )

    print("\nSample of optimized parameters:")
    for name, value in param_store.items():
        print(f"{name}: {value}")
