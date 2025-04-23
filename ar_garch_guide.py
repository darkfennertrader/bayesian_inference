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
    Hierarchical full-covariance guide:
    - Each asset's latent vector mean is shrunk toward a global latent mean.
    - All assets share a global full-covariance.
    """
    device = kwargs.get("device", torch.device("cpu"))
    ############################################################################
    print(
        "GUIDE() DEBUG: device arg:", device, "returns.device:", returns.device
    )
    ############################################################################

    indices = kwargs.get("indices", None)
    assert indices is not None, "Indices must be passed to the guide!"
    batch_size, _ = returns.shape
    param_dim = 7

    #########################################################################
    print(
        "GUIDE() | returns.shape:", returns.shape, "| batch_size:", batch_size
    )
    #########################################################################

    # ==== GLOBAL HYPERPRIORS (unchanged, copy-paste as in your setup) ====
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

    # ==== HIERARCHICAL LOCAL PARAMETERS ====
    # Global latent mean and covariance for all assets
    global_loc = pyro.param("global_loc", torch.zeros(param_dim, device=device))
    global_scale_tril = pyro.param(
        "global_scale_tril",
        torch.eye(param_dim, device=device),
        constraint=constraints.lower_cholesky,
    )

    # Asset-specific offsets (mean-reverting to global mean)
    # local_offset_all is shape [num_assets, param_dim]
    n = indices.shape[0] if isinstance(indices, torch.Tensor) else 1
    local_offset_all = pyro.param(
        "local_offset",
        torch.zeros(n, param_dim, device=device),
    )
    # Select only those in current batch:
    local_offset = local_offset_all[indices]  # [batch_size, param_dim]
    local_loc = global_loc + local_offset  # [batch_size, param_dim]
    ###################################################################
    print(
        "GUIDE() DEBUG: local_offset.shape",
        local_offset.shape,
        "local_offset.device",
        local_offset.device,
        "local_loc.device",
        local_loc.device,
    )
    ###################################################################

    ##############################################################################
    print("GUIDE() | about to enter assets plate, batch_size =", batch_size)
    print(
        "GUIDE() DEBUG: batching local_loc.device",
        local_loc.device,
        "global_scale_tril.device",
        global_scale_tril.device,
    )
    ##############################################################################

    with pyro.plate("assets", batch_size, dim=-2, device=device):
        zs = pyro.sample(
            "local_packed",
            dist.MultivariateNormal(local_loc, scale_tril=global_scale_tril),
            infer={"is_auxiliary": True},
        )
        #####################################################################
        print(
            "GUIDE() | zs.shape (local_packed):", zs.shape, zs.device, zs.device
        )
        ####################################################################

        # Unpack each variable for explicit guide matching:
        garch_omega_val = torch.exp(zs[..., 0])  # positive
        alpha_beta_sum_val = torch.sigmoid(zs[..., 1])  # (0,1)
        alpha_frac_val = torch.sigmoid(zs[..., 2])  # (0,1)
        phi_val = zs[..., 3]  # real
        garch_sigma_init_val = torch.exp(zs[..., 4])  # positive
        degrees_of_freedom_raw_val = torch.exp(
            zs[..., 5]
        )  # positive, model adds 2
        asset_weight_val = torch.sigmoid(zs[..., 6])  # (0,1)

        ###################################################################
        print(
            "GUIDE() DEBUG:",
            [
                (name, p.device)
                for name, p in [
                    ("garch_omega_val", garch_omega_val),
                    ("alpha_beta_sum_val", alpha_beta_sum_val),
                    ("alpha_frac_val", alpha_frac_val),
                    ("phi_val", phi_val),
                    ("garch_sigma_init_val", garch_sigma_init_val),
                    ("degrees_of_freedom_raw_val", degrees_of_freedom_raw_val),
                    ("asset_weight_val", asset_weight_val),
                ]
            ],
        )
        ###################################################################

        # Explicitly sample each with same name as model (using Delta to avoid warnings)
        pyro.sample("garch_omega", dist.Delta(garch_omega_val))
        pyro.sample("alpha_beta_sum", dist.Delta(alpha_beta_sum_val))
        pyro.sample("alpha_frac", dist.Delta(alpha_frac_val))
        pyro.sample("phi", dist.Delta(phi_val))
        pyro.sample("garch_sigma_init", dist.Delta(garch_sigma_init_val))
        pyro.sample(
            "degrees_of_freedom_raw", dist.Delta(degrees_of_freedom_raw_val)
        )
        pyro.sample("asset_weight", dist.Delta(asset_weight_val))

        #######################################################################
        print(
            f"GUIDE() | garch_omega_val.shape:{garch_omega_val.shape}, device:{garch_omega_val.device}"
        )
        print(
            f"GUIDE() | alpha_beta_sum_val.shape:{alpha_beta_sum_val.shape}, device:{alpha_beta_sum_val.device}"
        )
        print(
            f"GUIDE() | alpha_frac_val.shape:{alpha_frac_val.shape}, , device:{alpha_frac_val.device}"
        )
        print(
            f"GUIDE() | phi_val.shape:{phi_val.shape}, device:{phi_val.device}"
        )
        print(
            f"GUIDE() | garch_sigma_init_val.shape:{garch_sigma_init_val.shape}, device:{garch_sigma_init_val.device}"
        )
        print(
            f"GUIDE() | degrees_of_freedom_raw_val.shape:{degrees_of_freedom_raw_val.shape}, device:{degrees_of_freedom_raw_val.device}"
        )
        print(
            f"GUIDE() | asset_weight_val.shape:{asset_weight_val.shape}, device:{asset_weight_val.device}"
        )


##########################################################################


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
        model_args=(dummy_returns, dummy_lengths),
        model_kwargs={"args": args},
        filename="ar_garch_StudT_multiassets_partialpool_guide.jpg",
        render_params=True,
        render_distributions=True,
        # render_deterministic=True,
    )
