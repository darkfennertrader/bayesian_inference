# pylint: disable=no-name-in-module
from math import nan
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam  # type: ignore
from pyro.util import ignore_jit_warnings
import matplotlib.pyplot as plt
import seaborn as sns


pyro.set_rng_seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


# Helper: run model and guide as before
def model(data, half_life_tensor, device):
    T, N = data.shape
    decay_rate = (
        torch.log(torch.full((N,), 2.0, device=device)) / half_life_tensor
    )
    with pyro.plate("assets", N):
        z_prev = pyro.sample(
            "z_0",
            dist.Normal(
                torch.zeros(N, device=device), torch.ones(N, device=device)
            ),
        )
        for t in pyro.markov(range(T)):
            z_t = pyro.sample(
                f"z_{t+1}", dist.Normal(z_prev, torch.ones(N, device=device))
            )
            obs = data[t]
            mask = ~torch.isnan(obs)
            pyro.sample(f"x_{t+1}", dist.Normal(z_t[mask], 0.5), obs=obs[mask])
            penalty = -1.0 * torch.exp(-decay_rate * t)
            penalty_masked = penalty[mask]
            pyro.factor(f"penalty_t{t+1}", penalty_masked.sum())
            z_prev = z_t


def guide(data, half_life_tensor, device):
    T, N = data.shape
    locs = pyro.param("locs", torch.zeros(T + 1, N, device=device))
    scales = pyro.param(
        "scales",
        torch.ones(T + 1, N, device=device),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("assets", N):
        for t in range(T + 1):
            pyro.sample(f"z_{t}", dist.Normal(locs[t, :], scales[t, :]))


def run_svi(model, guide, loss_type, steps, data, half_life_tensor, device):
    pyro.clear_param_store()
    optimizer = Adam({"lr": 0.05})
    svi = SVI(
        lambda data, hl: model(data, hl, device),
        lambda data, hl: guide(data, hl, device),
        optimizer,
        loss=loss_type,
    )
    for _ in range(steps):
        _ = svi.step(data, half_life_tensor)
    # Output param snapshot (cpu for easy plotting)
    return {
        "locs": pyro.param("locs").detach().cpu().clone(),
        "scales": pyro.param("scales").detach().cpu().clone(),
    }


def batch_inference_all_assets(
    full_data, half_lives, device, n_steps=100, batch_size=2
):
    T, N = full_data.shape
    n_batches = (N + batch_size - 1) // batch_size
    all_params_trace = []
    all_params_jit = []
    for i in range(n_batches):
        asset_idx = slice(i * batch_size, min((i + 1) * batch_size, N))
        data_batch = full_data[:, asset_idx]
        hl_batch = half_lives[asset_idx]
        params_trace = run_svi(
            model, guide, Trace_ELBO(), n_steps, data_batch, hl_batch, device
        )
        all_params_trace.append(params_trace)
        try:
            params_jit = run_svi(
                model,
                guide,
                JitTrace_ELBO(),
                n_steps,
                data_batch,
                hl_batch,
                device,
            )
            all_params_jit.append(params_jit)
        except RuntimeError as e:  # This is appropriate here
            print(f"Batch {i+1} JitTrace_ELBO error:", str(e))
            all_params_jit.append(None)
    return all_params_trace, all_params_jit


# Stacked, labeled multi-batch summary figure (beautiful for presentations!)
def visualize_all_batches_summary(all_params_trace, all_params_jit):
    n_batches = len(all_params_trace)
    # 4 columns: Trace locs, JIT locs, loc diff, scale diff
    fig, axs = plt.subplots(n_batches, 4, figsize=(18, 5 * n_batches))
    for i, (trace, jit) in enumerate(zip(all_params_trace, all_params_jit)):
        row_axs = axs[i, :] if n_batches > 1 else axs
        # Plot Trace_ELBO locs
        sns.heatmap(
            trace["locs"],
            annot=True,
            fmt="0.3f",
            cmap="Blues",
            ax=row_axs[0],
            cbar=i == 0,
        )
        row_axs[0].set_title(f"Batch {i+1}: Trace_ELBO locs")
        # Plot JitTrace_ELBO locs or blank if failed
        if jit is not None:
            sns.heatmap(
                jit["locs"],
                annot=True,
                fmt="0.3f",
                cmap="Reds",
                ax=row_axs[1],
                cbar=i == 0,
            )
            row_axs[1].set_title(f"Batch {i+1}: JIT locs")
            # Difference LOCS
            diff_locs = trace["locs"] - jit["locs"]
            sns.heatmap(
                diff_locs,
                annot=True,
                fmt="0.3f",
                cmap="coolwarm",
                ax=row_axs[2],
                cbar=i == 0,
                center=0,
            )
            row_axs[2].set_title("LOC Diff (Trace − JIT)")
            # Difference SCALES
            diff_scales = trace["scales"] - jit["scales"]
            sns.heatmap(
                diff_scales,
                annot=True,
                fmt="0.3f",
                cmap="coolwarm",
                ax=row_axs[3],
                cbar=i == 0,
                center=0,
            )
            row_axs[3].set_title("SCALE Diff (Trace − JIT)")
        else:
            row_axs[1].set_axis_off()
            row_axs[2].set_axis_off()
            row_axs[3].set_axis_off()
        for ax in row_axs:
            ax.set_xlabel("asset in batch")
            ax.set_ylabel("time step")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.suptitle(
        "Bayesian Inference - Batched Comparison (Trace vs JIT and Diffs)",
        fontsize=18,
    )
    plt.show()


# --- Demo: Simulate and visualize

if __name__ == "__main__":
    import warnings

    # warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Simulate 4 assets x 3 time, with some NaNs, 2 assets per batch
    torch.manual_seed(7)
    full_data = torch.randn(3, 4) * 0.1
    full_data[0, 1] = float("nan")
    full_data[2, 3] = float("nan")
    full_data = full_data.to(device)
    print("\n", full_data, "\n")
    half_lives = torch.tensor([4.5, 7.0, 3.2, 6.7], device=device)
    all_params_trace, all_params_jit = batch_inference_all_assets(
        full_data, half_lives, device, n_steps=100, batch_size=2
    )
    visualize_all_batches_summary(all_params_trace, all_params_jit)
