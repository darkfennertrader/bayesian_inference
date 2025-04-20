# pylint: disable=not-callable

from dataclasses import dataclass
import torch
from torch.nn.functional import softplus
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints


def ar_garch_guide_student_t_multi_asset_partial_pooling(
    returns, lengths, **kwargs
):
    """
    Guide: strict parameter packing as in model.
    Transform local_params for correct support!
    """
    args = kwargs.get("args", {})
    prior_predictive_checks = kwargs.get("prior_predictive_checks", False)
    device = kwargs.get("device", torch.device("cpu"))
    num_assets = returns.shape[0]
    per_asset_param_dim = 7

    # ==== GLOBALS: lognormals, beta ====
    pyro.sample(
        "omega_mu",
        dist.LogNormal(
            pyro.param("omega_mu_loc"), pyro.param("omega_mu_scale")
        ),
    )
    pyro.sample(
        "omega_sigma",
        dist.LogNormal(
            pyro.param("omega_sigma_loc"), pyro.param("omega_sigma_scale")
        ),
    )
    pyro.sample(
        "ab_sum_a_hyper",
        dist.LogNormal(
            pyro.param("ab_sum_a_hyper_loc"), pyro.param("ab_sum_a_hyper_scale")
        ),
    )
    pyro.sample(
        "ab_sum_b_hyper",
        dist.LogNormal(
            pyro.param("ab_sum_b_hyper_loc"), pyro.param("ab_sum_b_hyper_scale")
        ),
    )
    pyro.sample(
        "ab_frac_a_hyper",
        dist.LogNormal(
            pyro.param("ab_frac_a_hyper_loc"),
            pyro.param("ab_frac_a_hyper_scale"),
        ),
    )
    pyro.sample(
        "ab_frac_b_hyper",
        dist.LogNormal(
            pyro.param("ab_frac_b_hyper_loc"),
            pyro.param("ab_frac_b_hyper_scale"),
        ),
    )
    pyro.sample(
        "phi_mu",
        dist.Normal(pyro.param("phi_mu_loc"), pyro.param("phi_mu_scale")),
    )
    pyro.sample(
        "phi_sigma",
        dist.LogNormal(
            pyro.param("phi_sigma_loc"), pyro.param("phi_sigma_scale")
        ),
    )
    pyro.sample(
        "sigma_init_mu",
        dist.LogNormal(
            pyro.param("sigma_init_mu_loc"), pyro.param("sigma_init_mu_scale")
        ),
    )
    pyro.sample(
        "sigma_init_sigma",
        dist.LogNormal(
            pyro.param("sigma_init_sigma_loc"),
            pyro.param("sigma_init_sigma_scale"),
        ),
    )
    pyro.sample(
        "df_mu",
        dist.LogNormal(pyro.param("df_mu_loc"), pyro.param("df_mu_scale")),
    )
    pyro.sample(
        "df_sigma",
        dist.LogNormal(
            pyro.param("df_sigma_loc"), pyro.param("df_sigma_scale")
        ),
    )
    pyro.sample(
        "lambda_decay",
        dist.Beta(
            pyro.param("lambda_decay_alpha"), pyro.param("lambda_decay_beta")
        ),
    )

    # ==== LOCAL FULL-COVARIANCE MVN ====
    local_loc = pyro.param("local_loc")  # [num_assets, 7]
    local_scale_tril = pyro.param(
        "local_scale_tril"
    )  # [num_assets, 7, 7], lower-triangular
    with pyro.plate("assets", num_assets):
        zs = pyro.sample(
            "local_params",
            dist.MultivariateNormal(local_loc, scale_tril=local_scale_tril),
        )

        # UNPACK GUIDE VECTOR FOR MODEL SUPPORT:
        garch_omega = torch.exp(zs[..., 0])  # positive
        alpha_beta_sum = torch.sigmoid(zs[..., 1])  # in (0,1)
        alpha_frac = torch.sigmoid(zs[..., 2])  # in (0,1)
        phi = zs[..., 3]  # real
        garch_sigma_init = torch.exp(zs[..., 4])  # positive
        degrees_of_freedom = 2.0 + softplus(zs[..., 5])  # >2
        asset_weight = torch.sigmoid(zs[..., 6])  # in (0,1)
        # These would be "repacked" as needed for downstream deterministic guides/amortized use.


if __name__ == "__main__":

    num_assets = 1
    per_asset_param_dim = 7

    # Register guide parameters (with good default initializations)
    pyro.clear_param_store()
    pyro.param("omega_mu_loc", torch.tensor(0.0))
    pyro.param("omega_mu_scale", torch.tensor(0.3))
    pyro.param("omega_sigma_loc", torch.tensor(0.0))
    pyro.param("omega_sigma_scale", torch.tensor(0.1))
    pyro.param("ab_sum_a_hyper_loc", torch.tensor(0.2))
    pyro.param("ab_sum_a_hyper_scale", torch.tensor(0.2))
    pyro.param("ab_sum_b_hyper_loc", torch.tensor(0.2))
    pyro.param("ab_sum_b_hyper_scale", torch.tensor(0.2))
    pyro.param("ab_frac_a_hyper_loc", torch.tensor(0.2))
    pyro.param("ab_frac_a_hyper_scale", torch.tensor(0.2))
    pyro.param("ab_frac_b_hyper_loc", torch.tensor(0.2))
    pyro.param("ab_frac_b_hyper_scale", torch.tensor(0.2))
    pyro.param("phi_mu_loc", torch.tensor(0.0))
    pyro.param("phi_mu_scale", torch.tensor(1.0))
    pyro.param("phi_sigma_loc", torch.tensor(0.0))
    pyro.param("phi_sigma_scale", torch.tensor(0.5))
    pyro.param("sigma_init_mu_loc", torch.tensor(2.0))
    pyro.param("sigma_init_mu_scale", torch.tensor(0.3))
    pyro.param("sigma_init_sigma_loc", torch.tensor(0.0))
    pyro.param("sigma_init_sigma_scale", torch.tensor(0.1))
    pyro.param("df_mu_loc", torch.tensor(1.0))
    pyro.param("df_mu_scale", torch.tensor(0.2))
    pyro.param("df_sigma_loc", torch.tensor(0.0))
    pyro.param("df_sigma_scale", torch.tensor(0.1))
    pyro.param("lambda_decay_alpha", torch.tensor(2.0))
    pyro.param("lambda_decay_beta", torch.tensor(2.0))

    pyro.param("local_loc", torch.zeros(num_assets, per_asset_param_dim))
    pyro.param(
        "local_scale_tril",
        torch.stack(
            [torch.eye(per_asset_param_dim) for _ in range(num_assets)]
        ),
        constraint=constraints.lower_cholesky,
    )
    # Dummy data with one asset, 3 time steps
    dummy_returns = torch.zeros(1, 3)
    dummy_lengths = torch.tensor([3])

    @dataclass
    class Args:
        jit: bool = False

    args = Args(jit=False)

    print("Rendering GUIDE structure...")

    pyro.render_model(
        ar_garch_guide_student_t_multi_asset_partial_pooling,
        model_args=(dummy_returns, dummy_lengths, args),
        filename="ar_garch_StudT_multiassets_partialpool_guide.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
