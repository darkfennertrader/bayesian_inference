import torch
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro import poutine


def hierarchical_dirichlet_multinomial_model(data):
    """
    data: list of length G, where data[g] is a 1D tensor of observed category indices for group g

    This example hard-codes alpha for a 4-category Dirichlet to demonstrate
    how to show alpha as a node in the graphical model.
    """
    # Make alpha a pyro.param so it appears in the graph, and then "promote" it to a
    # deterministic node called 'alpha'.
    alpha_param = pyro.param("alpha_param", torch.tensor([1.0, 1.0, 1.0, 1.0]))
    alpha = pyro.deterministic("alpha", alpha_param)

    # Number of groups (each group will get its own theta[g] parameter)
    num_groups = len(data)

    # For each group, sample a separate Dirichlet-distributed parameter vector theta[g]
    with pyro.plate("groups", num_groups):
        theta = pyro.sample("theta", dist.Dirichlet(alpha))

    # For each group g, observe each data point x[g][j] ~ Categorical(theta[g])
    for g, obs_g in enumerate(data):
        with pyro.plate(f"obs_{g}", len(obs_g)):
            pyro.sample(f"x_{g}", dist.Categorical(theta[g]), obs=obs_g)


def hierarchical_dirichlet_multinomial_hyper_model(data, K=4):
    """
    data: list of length G, where data[g] is a 1D tensor of observed category indices for group g.
    K   : number of discrete categories (defaults to 4).

    Model structure (plate notation):
      1) alpha ~ Gamma(...)^K  [a K-dimensional vector of Dirichlet concentration parameters]
      2) For each group g     : theta[g] ~ Dirichlet(alpha)
      3) For each data point j in group g: x[g][j] ~ Categorical(theta[g])

    This allows "partial pooling":
      - alpha is inferred from *all* groups' data,
      - each group has its own theta[g],
      - but all theta[g]s share a common prior alpha.
    """

    # -------------------------
    # 1) Sample the hyperparameters alpha (size K)
    #    Here, we put a (Gamma(2.0,1.0)) prior on each component of alpha.
    #    The "to_event(1)" ensures we treat alpha as a single K-dim random vector.
    # -------------------------
    alpha = pyro.sample("alpha", dist.Gamma(2.0, 1.0).expand([K]).to_event(1))

    # -------------------------
    # 2) Sample theta[g] for each group g
    # -------------------------
    num_groups = len(data)
    with pyro.plate("groups", num_groups):
        theta = pyro.sample("theta", dist.Dirichlet(alpha))

    # -------------------------
    # 3) For each observation x[g][j], within each group g:
    #    x[g][j] ~ Categorical(theta[g])
    # -------------------------
    for g, obs_g in enumerate(data):
        with pyro.plate(f"obs_{g}", len(obs_g)):
            pyro.sample(f"x_{g}", dist.Categorical(theta[g]), obs=obs_g)


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # 1. Suppose we have G=3 groups, each with different numbers of observations.
    data = [
        torch.tensor([0, 1, 1, 2]),  # group 0
        torch.tensor([1, 1, 0]),  # group 1
        torch.tensor([2, 2, 3]),  # group 2
    ]

    # 3. To render the model graphically in a notebook:

    graph = pyro.render_model(
        model=hierarchical_dirichlet_multinomial_hyper_model,
        model_args=(data,),
        render_distributions=True,
        render_deterministic=True,
    )
    graph.view()
