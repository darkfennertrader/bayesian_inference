from typing import Optional
import torch
import pyro
from pyro.distributions import constraints
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO

# pylint: disable=no-name-in-module
from pyro.optim import Adam  #  type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from ar_garch_model import ar_garch_model_student_t_multi_asset_partial_pooling
from ar_garch_guide import ar_garch_guide_student_t_multi_asset_partial_pooling


class AR_GARCH_SVIEngine:
    def __init__(self, model, guide, num_assets, param_dim, device):
        self.model = model
        self.guide = guide
        self.num_assets = num_assets
        self.param_dim = param_dim
        self.device = device
        self.optimizer = Adam({"lr": 1e-3})
        self.svi: Optional[SVI] = None
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
            "local_loc",
            torch.zeros(self.num_assets, self.param_dim, device=self.device),
        )
        pyro.param(
            "local_scale_tril",
            torch.stack(
                [
                    torch.eye(self.param_dim, device=self.device)
                    for _ in range(self.num_assets)
                ]
            ),
            constraint=constraints.lower_cholesky,
        )

    def build_svi(self, use_jit=True):
        elbo = JitTrace_ELBO() if use_jit else Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=elbo)

    def run(
        self,
        returns,
        lengths,
        num_epochs=50,
        batch_size=16,
        args=None,
        prior_predictive_checks=False,
    ):
        args = args or {}
        self.initialize_params()
        self.build_svi()
        num_assets = returns.shape[0]
        num_batches = (num_assets + batch_size - 1) // batch_size
        # JIT warmup
        if self.svi is None:
            raise RuntimeError("SVI must be built before usage.")
        print("Running JIT warmup...")
        self.svi.guide(
            returns[:batch_size],
            lengths[:batch_size],
            args=args,
            prior_predictive_checks=prior_predictive_checks,
            device=self.device,
        )

        for epoch in range(num_epochs):
            perm = torch.randperm(num_assets)
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_assets)
                indices = perm[start:end].to(device=self.device)
                batch_returns = returns[indices]
                batch_lengths = lengths[indices]
                loss = self.svi.step(
                    batch_returns,
                    batch_lengths,
                    args=args,
                    prior_predictive_checks=prior_predictive_checks,
                    device=self.device,
                )
                epoch_loss += float(loss)  # type: ignore
            print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.3f}")


if __name__ == "__main__":
    num_assets = 2
    per_asset_param_dim = 7
    max_T = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    returns = torch.randn(num_assets, max_T, device=device)
    lengths = torch.full((num_assets,), max_T, dtype=torch.long, device=device)

    pyro.set_rng_seed(0)
    pyro.enable_validation(True)
    engine = AR_GARCH_SVIEngine(
        ar_garch_model_student_t_multi_asset_partial_pooling,
        ar_garch_guide_student_t_multi_asset_partial_pooling,
        num_assets,
        per_asset_param_dim,
        device,
    )

    engine.run(returns, lengths, num_epochs=50, batch_size=2)
