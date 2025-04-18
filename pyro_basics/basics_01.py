# pylint: disable=W0105
import os
import random
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.distributions import Categorical, MultivariateNormal, Normal
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine
from pyro.optim import Adam  # pylint: disable=no-name-in-module # type: ignore


smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.9.1")
# pylint: disable=W0105


# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)


# can line up trailing dimensions to make reading easier but not necessary:
# x = torch.ones(1)
# print(x)
# torch.manual_seed(42)
# y = torch.randint(low=0, high=9, size=(3, 1, 7))
# print(y)
# z = x + y
# print()
# print((z))
# print(z.shape)

"""
In PyTorch (and in Pyro), a Distribution object manages two concepts of shape:

• batch_shape: This echoes how many independent distributions are being considered at once (the “batch” dimensions).
• event_shape: This corresponds to the dimensionality of an individual sample (the “event” dimensions).

So the total shape of a sample drawn from a Distribution is batch_shape + event_shape.
"""
# 1) A univariate Normal distribution with a non-empty batch_shape
# Consider a univariate Normal distribution with a batch_shape of (2, 3),
# meaning we have 2×3 = 6 distinct Normal distributions in parallel.
# We can create it in Pyro roughly as:

# Loc and scale each have shape (2,3).
loc = torch.zeros(2, 3)  # batch_shape = (2,3)
scale = torch.ones(2, 3)  # same shape

d = dist.Normal(loc, scale)
print("batch_shape =", d.batch_shape)  # (2,3)
print("event_shape =", d.event_shape)  # ()


print("\nDrawing a single sample (no sample_shape)")
x = d.sample()
print("Sample shape:", x.shape)  # (2,3)
print("Log prob shape:", d.log_prob(x).shape)  # (2,3)

print("\nDrawing multiple i.i.d. samples (sample_shape = (4,))")
x2 = d.sample(sample_shape=(4,))
print("x2 shape:", x2.shape)  # (4, 2, 3)
print("Log prob shape:", d.log_prob(x2).shape)  # (4, 2, 3)

"""
Explanation (Univariate Normal case): • event_shape is (), because each random
draw is univariate (a single scalar). • batch_shape is (2,3), meaning we have 6
parallel/scalar distributions. • When we sample with sample_shape=(4,), we get
(4, 2, 3). Those 4 are i.i.d. samples (sample_shape). The 2 and 3 come from
batch_shape.

Since event_shape is empty, log_prob returns one log probability per batch
element (and per sample, if sample_shape was used). Hence log_prob(x2).shape is
(4, 2, 3)."""
"""

2. A multivariate Normal distribution with a non-empty event_shape
Now consider a 3-dimensional multivariate Normal distribution. By default, that
has no batch_shape (i.e. ()), but it has event_shape = (3,). One way to create
it is:

"""

loc3 = torch.zeros(3)
cov3 = torch.eye(3)  # 3x3 identity
d3 = dist.MultivariateNormal(loc3, covariance_matrix=cov3)

print("\n--- 3-dimensional MULTIVARIATE ---")
x3 = d3.sample()
print("Sample shape:", x3.shape)  # (3,)
print("Log prob shape:", d3.log_prob(x3).shape)  # ()

x3_multi = d3.sample((5,))
print("\nx3_multi shape:", x3_multi.shape)  # (5, 3)
print("Log prob shape:", d3.log_prob(x3_multi).shape)  # (5,)

"""
Explanation (Multivariate Normal case): • batch_shape is (), meaning we do not
have separate parallel distributions. • event_shape is (3,), meaning each single
draw is a 3-dimensional vector that is jointly distributed (dependent). • When
we take a single sample, we get shape (3,). That is one vector in R^3. • When we
compute log_prob of that single sample x3, we get a single scalar output (shape
()). All 3 coordinates together form one "event," so the log probability is for
the entire event. • If we draw multiple samples by setting sample_shape = (5,),
the shape is (5, 3). Those are 5 i.i.d. draws, each in R^3. The log
probabilities then have shape (5,), returning one scalar per sample.
"""


"""
3. Putting it all together

In general, when you call d.sample(sample_shape), PyTorch merges the shapes as:
sample_shape + batch_shape + event_shape

Indices along sample_shape correspond to i.i.d. draws of the entire
distribution. Indices along batch_shape correspond to conditionally independent
distributions (each with separate parameters). Indices along event_shape
correspond to the dimension over which a single draw is jointly dependent (e.g.
for multivariate distributions).

Hence: • d.sample() has shape = batch_shape + event_shape if no sample_shape is
provided. • d.log_prob(x) sums (or integrates) over the event_shape, returning
shape = sample_shape + batch_shape if x has shape sample_shape + batch_shape +
event_shape.

This distinction helps manage complicated models in probabilistic programming,
especially when stacking distributions of different dimensions or when dealing
with time-series or hierarchical models (e.g. in Pyro, Tensor shapes must match
these patterns so that .log_prob() calculations work as intended).
"""


# Another example:
def main():
    # Means: shape [3, 2] => batch_shape is [3], event_shape is [2]
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    # Covariance matrices: shape [3, 2, 2] => keeps batch_shape as [3], event_shape as [2]
    cov_matrices = torch.stack([torch.eye(2), 2.0 * torch.eye(2), 3.0 * torch.eye(2)], dim=0)
    print(cov_matrices)

    # Create the multivariate normal distribution
    mvn = dist.MultivariateNormal(loc=means, covariance_matrix=cov_matrices)

    # Now let's print out batch_shape and event_shape directly from the distribution
    print("\nDistribution batch_shape:", mvn.batch_shape)
    print("Distribution event_shape:", mvn.event_shape)

    # We'll sample from the distribution with sample_shape = [4].
    samples = mvn.sample(sample_shape=(4,))

    # Print sample shape explicitly as we set it:
    print("Sample shape used:", (4,))

    # Print the final tensor shape after sampling:
    print("Resulting samples shape:", samples.shape)


# main()
print()
print(torch.zeros(2))


def model1():
    a = pyro.sample("a", Normal(0, 1))
    b = pyro.sample("b", Normal(torch.zeros(2), 1).to_event(1))
    with pyro.plate("c_plate", 2):
        c = pyro.sample("c", Normal(torch.zeros(2), 1))
    with pyro.plate("d_plate", 3):
        d = pyro.sample("d", Normal(torch.zeros(3, 4, 5), 1).to_event(2))
    assert a.shape == ()  # batch_shape == ()     event_shape == ()
    assert b.shape == (2,)  # batch_shape == ()     event_shape == (2,)
    assert c.shape == (2,)  # batch_shape == (2,)   event_shape == ()
    assert d.shape == (3, 4, 5)  # batch_shape == (3,)   event_shape == (4,5)

    x_axis = pyro.plate("x_axis", 3, dim=-2)
    y_axis = pyro.plate("y_axis", 2, dim=-3)
    with x_axis:
        x = pyro.sample("x", Normal(0, 1))
    with y_axis:
        y = pyro.sample("y", Normal(0, 1))
    with x_axis, y_axis:
        xy = pyro.sample("xy", Normal(0, 1))
        z = pyro.sample("z", Normal(0, 1).expand([5]).to_event(1))
    assert x.shape == (3, 1)  # batch_shape == (3,1)     event_shape == ()
    assert y.shape == (2, 1, 1)  # batch_shape == (2,1,1)   event_shape == ()
    assert xy.shape == (2, 3, 1)  # batch_shape == (2,3,1)   event_shape == ()
    assert z.shape == (2, 3, 1, 5)  # batch_shape == (2,3,1)   event_shape == (5,)


print("\nMODEL 1:")
test_model(model1, guide=lambda: None, loss=Trace_ELBO())

trace = poutine.trace(model1).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
