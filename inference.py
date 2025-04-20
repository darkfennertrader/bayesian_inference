import torch
import pyro
from pyro.distributions import constraints
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO

# pylint: disable=no-name-in-module
from pyro.optim import Adam  #  type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from ar_garch_model import ar_garch_model_student_t_multi_asset_partial_pooling
from ar_garch_guide import guide

num_assets = 52
per_asset_param_dim = 7  # << Only defined here, shared by model/guide
max_T = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 50

returns = torch.randn(num_assets, max_T, device=device)
lengths = torch.full((num_assets,), max_T, dtype=torch.long, device=device)

# ----- PARAMETER REGISTRATION: Variational Params (locals only) ----
pyro.clear_param_store()
pyro.param(
    "local_loc", torch.zeros(num_assets, per_asset_param_dim, device=device)
)
pyro.param(
    "local_scale_tril",
    torch.stack(
        [
            torch.eye(per_asset_param_dim, device=device)
            for _ in range(num_assets)
        ]
    ),
    constraint=constraints.lower_cholesky,
)

optimizer = Adam({"lr": 1e-3})
svi = SVI(
    ar_garch_model_student_t_multi_asset_partial_pooling,
    guide,
    optimizer,
    loss=Trace_ELBO(),
)


def run_svi_minibatch(
    svi,
    returns,
    lengths,
    num_epochs,
    batch_size,
    device,
    args,
    prior_predictive_checks=False,
):
    num_assets = returns.shape[0]
    num_batches = (num_assets + batch_size - 1) // batch_size  # ceil division
    for epoch in range(num_epochs):
        perm = torch.randperm(num_assets)  # always on CPU by default
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_assets)
            indices = perm[start:end].to(device=device)
            batch_returns = returns[indices]
            batch_lengths = lengths[indices]
            loss = svi.step(
                batch_returns,
                batch_lengths,
                args,
                prior_predictive_checks=prior_predictive_checks,
                device=device,
            )
            epoch_loss += loss
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.3f}")


print("Training SVI minibatch model...")
run_svi_minibatch(
    svi,
    returns,
    lengths,
    num_epochs=num_epochs,
    batch_size=batch_size,
    device=device,
    args={},
    prior_predictive_checks=False,
)
print("Done!")


if __name__ == "__main__":
    pass
