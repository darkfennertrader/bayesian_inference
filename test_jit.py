# pylint: disable=no-name-in-module
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam  # type: ignore
from pyro.infer.autoguide import AutoDiagonalNormal


# Dummy model that branches on length and mode
def asset_model(sequence, length, mode="fast"):
    loc = pyro.sample("loc", dist.Normal(0.0, 1.0))
    scale = pyro.sample("scale", dist.LogNormal(0.0, 0.3))
    with pyro.plate("data", length):
        # Dummy use of mode (could be some structural change here!)
        if mode == "fast":
            pyro.sample("obs", dist.Normal(loc, scale), obs=sequence)
        else:
            pyro.sample(
                "obs", dist.Normal(loc * 2, scale), obs=sequence
            )  # STRUCTURE VARIES


# Guide for the model
guide = AutoDiagonalNormal(asset_model)

# Fake data for two assets
asset_A = torch.randn(24)
asset_B = torch.randn(36)

# Prepare
pyro.clear_param_store()
guide(asset_A, length=24, mode="fast")  # warmup, avoids lazy init side effects
elbo = JitTrace_ELBO()
svi = SVI(asset_model, guide, Adam({"lr": 0.01}), elbo)

# All (length, mode) combinations
lengths = [24, 36]
modes = ["fast", "slow"]
assets = [asset_A, asset_B]

for sequence, length in zip(assets, lengths):
    for mode in modes:
        print(f"\nTraining on sequence of length={length}, mode={mode}")
        svi.step(sequence, length=length, mode=mode)
