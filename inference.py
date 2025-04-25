from typing import Optional
import torch
import pyro
from pyro.distributions import constraints
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO

# pylint: disable=no-name-in-module
from pyro.optim import Adam  #  type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from pyro_model import ar_garch_model_student_t_multi_asset_partial_pooling
from pyro_guide import ar_garch_guide_student_t_multi_asset_partial_pooling
from helpers import debug_shape


# ----- Strategy/Interface -----
class SVIStrategy:
    def step(self, *args, **kwargs):
        raise NotImplementedError


# ----- Concrete Strategies -----
class JITSVIStrategy(SVIStrategy):
    def __init__(self, model, guide, optimizer, batch_size, device):
        self.svi = SVI(model, guide, optimizer, loss=JitTrace_ELBO())
        # Warm up JIT for correct batch size
        dummy_returns = torch.zeros(batch_size, 1, device=device)
        dummy_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
        ####################################################################
        print(
            "JIT WARMUP | dummy_returns.shape:",
            dummy_returns.shape,
            "| dummy_lengths.shape:",
            dummy_lengths.shape,
        )
        ####################################################################
        dummy_indices = torch.arange(batch_size, device=device)
        self.svi.guide(
            dummy_returns, dummy_lengths, indices=dummy_indices, device=device
        )

    def step(self, *args, **kwargs):
        ##################################################################
        print(
            "JITSVIStrategy.step: args devices:",
            [a.device if hasattr(a, "device") else "NA" for a in args],
        )
        ###################################################################
        return self.svi.step(*args, **kwargs)


class RegularSVIStrategy(SVIStrategy):
    def __init__(self, model, guide, optimizer):
        self.svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    def step(self, *args, **kwargs):
        #############################################################
        print(
            "RegularSVIStrategy.step: args devices:",
            [a.device if hasattr(a, "device") else "NA" for a in args],
        )
        ############################################################
        return self.svi.step(*args, **kwargs)


# ----- SVI Engine -----
class AR_GARCH_SVIEngine:
    def __init__(self, model, guide, num_assets, param_dim, max_T, device):
        self.model = model
        self.guide = guide
        self.num_assets = num_assets
        self.param_dim = param_dim
        self.device = device
        self.optimizer = Adam({"lr": 1e-3})
        self.jit_strategy = None
        self.regular_strategy = None
        self.max_T = max_T
        self.global_param_specs = [
            ("omega_mu_loc", 0.0, None),
            ("omega_mu_scale", 1.0, constraints.positive),
            ("omega_sigma_loc", 0.0, None),
            ("omega_sigma_scale", 1.0, constraints.positive),
            ("ab_sum_a_hyper_loc", 0.5, None),
            ("ab_sum_a_hyper_scale", 0.5, constraints.positive),
            ("ab_sum_b_hyper_loc", 0.5, None),
            ("ab_sum_b_hyper_scale", 0.5, constraints.positive),
            ("ab_frac_a_hyper_loc", 0.5, None),
            ("ab_frac_a_hyper_scale", 0.5, constraints.positive),
            ("ab_frac_b_hyper_loc", 0.5, None),
            ("ab_frac_b_hyper_scale", 0.5, constraints.positive),
            ("phi_mu_loc", 0.0, None),
            ("phi_mu_scale", 1.0, constraints.positive),
            ("phi_sigma_loc", 0.0, None),
            ("phi_sigma_scale", 1.0, constraints.positive),
            ("sigma_init_mu_loc", 2.0, None),
            ("sigma_init_mu_scale", 0.5, constraints.positive),
            ("sigma_init_sigma_loc", 0.0, None),
            ("sigma_init_sigma_scale", 0.5, constraints.positive),
            ("df_mu_loc", 1.0, None),
            ("df_mu_scale", 0.5, constraints.positive),
            ("df_sigma_loc", 0.0, None),
            ("df_sigma_scale", 0.5, constraints.positive),
            ("lambda_decay_alpha", 2.0, constraints.positive),
            ("lambda_decay_beta", 2.0, constraints.positive),
        ]

    def initialize_params(self):
        pyro.clear_param_store()
        for name, init_val, constraint in self.global_param_specs:
            tensor = torch.tensor(init_val, device=self.device)
            if constraint is not None:
                pyro.param(name, tensor, constraint=constraint)
            else:
                pyro.param(name, tensor)
        pyro.param(
            "local_offset",
            torch.zeros(self.num_assets, self.param_dim, device=self.device),
        )
        pyro.param(
            "global_loc",
            torch.zeros(self.param_dim, device=self.device),
        )
        pyro.param(
            "global_scale_tril",
            torch.eye(self.param_dim, device=self.device),
            constraint=constraints.lower_cholesky,
        )
        ##############################################################

        # Print all params and their devices
        param_devices = [
            (name, value.device)
            for name, value in pyro.get_param_store().items()
        ]
        # print("\nParameter device list:", param_devices)

        # Find out if any are on CPU
        any_on_cpu = any(device.type == "cpu" for (_, device) in param_devices)
        if any_on_cpu:
            print("WARNING: Some parameters are on CPU!")
        else:
            print("All parameters are on the GPU.")
        #############################################################

    def prepare_strategies(self, batch_size):
        self.jit_strategy = JITSVIStrategy(
            lambda *a, **kw: self.model(*a, **kw, max_T=self.max_T),
            lambda *a, **kw: self.guide(*a, **kw, max_T=self.max_T),
            self.optimizer,
            batch_size,
            self.device,
        )
        self.regular_strategy = RegularSVIStrategy(
            lambda *a, **kw: self.model(*a, **kw, max_T=self.max_T),
            lambda *a, **kw: self.guide(*a, **kw, max_T=self.max_T),
            self.optimizer,
        )

    def run(
        self,
        returns,
        lengths,
        num_epochs: int,
        batch_size: int,
        args=None,
        prior_predictive_checks=False,
    ):
        args = args or {}
        self.initialize_params()
        self.prepare_strategies(batch_size)
        num_assets = returns.shape[0]
        num_batches = (num_assets + batch_size - 1) // batch_size
        print(
            f"num_assets: {num_assets}, batch_size: {batch_size}, num_batches: {num_batches}"
        )

        for epoch in range(num_epochs):
            perm = torch.randperm(num_assets)
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_assets)
                indices = perm[start:end].to(device=self.device)
                print(indices, "type:", type(indices))
                batch_returns = returns[indices]
                batch_lengths = lengths[indices]
                current_batch_size = batch_returns.shape[0]

                # Choose strategy
                if current_batch_size == batch_size:
                    print(
                        "\nSVI step (Jit)| batch_returns.shape:",
                        batch_returns.shape,
                        "| batch_lengths.shape:",
                        batch_lengths.shape,
                        "| current_batch_size:",
                        current_batch_size,
                    )
                    svi_strategy = self.jit_strategy
                else:
                    print(
                        "\nSVI step (Regular)| batch_returns.shape:",
                        batch_returns.shape,
                        "| batch_lengths.shape:",
                        batch_lengths.shape,
                        "| current_batch_size:",
                        current_batch_size,
                    )
                    svi_strategy = self.regular_strategy

                # svi_strategy = (
                #     self.jit_strategy
                #     if current_batch_size == batch_size
                #     else self.regular_strategy
                # )

                ####################################################
                print(
                    f"INFERENCE DEBUG: batch_returns.device: {batch_returns.device}, batch_lengths.device: {batch_lengths.device}, indices.device: {indices.device}, device param: {self.device}"
                )
                ####################################################

                assert svi_strategy is not None
                loss = svi_strategy.step(
                    batch_returns,
                    batch_lengths,
                    indices=indices,
                    args=args,
                    prior_predictive_checks=prior_predictive_checks,
                    device=self.device,
                )

                assert isinstance(
                    loss, (float, int, torch.Tensor)
                ), f"Bad loss type: {type(loss)}"
                epoch_loss += float(loss)
                print(
                    f"[Epoch {epoch+1} Batch {batch_idx+1}] Batch size: {current_batch_size} | Loss: {loss:.3f}"
                )

            print(f"Epoch {epoch + 1} - Total Loss: {epoch_loss:.3f}")


if __name__ == "__main__":
    pyro.set_rng_seed(0)
    pyro.enable_validation(True)
    # Dummy model/guide definition for test below!
    num_assets = 2
    param_dim = 7
    max_T = 5
    batch_size = 1
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    returns = torch.randn(num_assets, max_T, device=device)
    lengths = torch.full((num_assets,), max_T, dtype=torch.long, device=device)
    ###################################################################
    print(
        "MAIN | returns.shape:",
        returns.shape,
        "| lengths.shape:",
        lengths.shape,
        "| batch_size:",
        batch_size,
    )
    ####################################################################
    engine = AR_GARCH_SVIEngine(
        ar_garch_model_student_t_multi_asset_partial_pooling,
        ar_garch_guide_student_t_multi_asset_partial_pooling,
        num_assets,
        param_dim,
        max_T,
        device,
    )
    engine.run(returns, lengths, num_epochs=num_epochs, batch_size=batch_size)
