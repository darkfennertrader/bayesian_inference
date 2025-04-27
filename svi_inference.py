from math import nan
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import random
import numpy as np
import torch
import pyro
from pyro.distributions import constraints
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO

# pylint: disable=no-name-in-module
from pyro.optim import Adam  #  type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from model_02 import ar_garch2_studentt_model
from guide_02 import ar_garch2_studentt_guide


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ElboType(str, Enum):
    JIT = "jit"
    TRACE = "trace"
    ENUM = "enum"


@dataclass
class SVIConfig:
    device: Device = Device.CPU
    elbo_type: ElboType = ElboType.JIT
    n_steps: int = 500
    lr: float = 0.01
    r: int = 3
    seed: int = 42
    verbose: bool = True
    # Add any future fields here, e.g., optional regularization, scheduler settings


class SVIInference:
    def __init__(self, model, guide, data):
        self.model = model
        self.guide = guide
        self.data = data
        self.loss_trace = []
        self.run_time = None
        self.results_dict = None

    def run(self, config: SVIConfig, **kwargs):
        # ---- Set reproducibility, device-dependent ----
        self._set_seeds(config)
        self.data = self.data.to(config.device.value)

        pyro.clear_param_store()
        elbo = self._get_elbo(config.elbo_type.value)
        optimizer = Adam({"lr": config.lr})
        svi = SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=elbo,
        )
        # Ensure device and r are in kwargs for model/guide
        kw = dict(kwargs)
        kw["device"] = config.device
        kw["r"] = config.r

        self.loss_trace = []
        t0 = time.time()
        for step in range(config.n_steps):
            loss = svi.step(self.data, **kw)
            if torch.isnan(torch.tensor(loss)):
                print("NaN loss at step:", step)
                break
            self.loss_trace.append(loss)
            if config.verbose and (
                step % max(1, config.n_steps // 10) == 0
                or step == config.n_steps - 1
            ):
                print(f"[{step+1:4d}/{config.n_steps}] Loss: {loss:.2f}")
        t1 = time.time()
        self.run_time = t1 - t0
        print(f"\nSVI run complete in {self.run_time:.3f} seconds.")

        self.results_dict = {
            "loss_trace": self.loss_trace,
            "elapsed": self.run_time,
            "pyro_params": {
                k: v.detach().cpu().numpy()
                for k, v in pyro.get_param_store().items()
            },
        }

    @property
    def results(self):
        if self.results_dict is None:
            raise RuntimeError("Call .run() first.")
        return self.results_dict

    def summary(self, topn=5):
        d = self.results
        print(f"\nFinal loss: {d['loss_trace'][-1]:.2f}")
        print(f"Elapsed: {d['elapsed']:.3f}s")
        print("Params:")
        for param in list(d["pyro_params"].keys())[:topn]:
            print(f"   {param}: {d['pyro_params'][param].shape}")
        if len(d["pyro_params"]) > topn:
            print("  ...")

    def _get_elbo(self, elbo_type):
        if elbo_type == "jit":
            return JitTrace_ELBO()
        elif elbo_type == "trace":
            return Trace_ELBO()
        elif elbo_type == "enum":
            return TraceEnum_ELBO()
        else:
            raise ValueError(f"Unknown ELBO type: {elbo_type}")

    def _set_seeds(self, config: SVIConfig):
        pyro.set_rng_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available!")
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ----- Dummy Data Example -----


def get_dummy_data(device):
    # Now shape is [T, N_assets]
    return torch.tensor(
        [
            [0.01, float("nan")],
            [0.00, 0.01],
            [float("nan"), 0.03],
        ],
        dtype=torch.float32,
        device=device,
    )


if __name__ == "__main__":
    # Define ar_garch2_studentt_model and ar_garch2_studentt_guide above this block

    # Config (adjust defaults as you like!)
    config = SVIConfig(
        device=Device.CUDA,
        elbo_type=ElboType.JIT,
        verbose=True,
    )

    # Data
    returns = get_dummy_data(config.device)
    svi = SVIInference(
        ar_garch2_studentt_model, ar_garch2_studentt_guide, returns
    )
    svi.run(config)
    svi.summary()
