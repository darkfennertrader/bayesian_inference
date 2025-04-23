import torch
import pyro
import pyro.distributions as dist


def obs_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = torch.tensor(2.0, device=device)
    beta = torch.tensor(2.0, device=device)
    val = torch.tensor(0.2, device=device)  # device mismatch!
    try:
        pyro.sample("beta_obs", dist.Beta(alpha, beta), obs=val)
    except Exception as e:
        print("Obs error:", e)


# obs_demo()


def beta_param_mismatch():
    pyro.clear_param_store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define params, but one is on CPU and one is on CUDA by mistake
    pyro.param(
        "alpha",
        torch.tensor(2.0, device=device),
        constraint=dist.constraints.positive,
    )
    pyro.param(
        "beta",
        torch.tensor(5.0, device="cuda"),
        constraint=dist.constraints.positive,
    )

    # Try to use these in a model
    def model():
        alpha = pyro.param("alpha")
        beta = pyro.param("beta")
        try:
            pyro.sample("x", dist.Beta(alpha, beta))
        except Exception as e:
            print("Inside model(): Error:", e)

    model()


beta_param_mismatch()
