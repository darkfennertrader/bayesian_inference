import os
import random
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.distributions import Categorical, MultivariateNormal, Normal, Bernoulli  # type: ignore
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate, SVI
import pyro.poutine as poutine
from pyro.optim import Adam  # pylint: disable=no-name-in-module # type: ignore


smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.9.1")
# pylint: disable=W0105

pyro.set_rng_seed(0)


# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)


# Below is a step‐by‐step explanation of why each sampled tensor ends up with
# the asserted shape under Pyro’s enumeration machinery, given:

# • a is a Categorical over 6 outcomes • b, c, d are Bernoulli variables (each
# enumerated to 2 outcomes) • c is inside a plate of size 4 (c_plate) • d is
# inside a nested plate of size 5 (d_plate) • e is a Normal with event dimension
# 1 (size 7), depending on whether d = 0 or d = 1

# a.shape == (6, 1, 1) • a ~ Categorical(...) with 6 possible outcomes. When
# enumerated, Pyro adds an extra leftmost dimension of size 6 to represent the 6
# enumerated values. • The remaining “(1, 1)” are placeholder/broadcast
# dimensions introduced because Pyro reserves space for potential downstream
# enumerations and plates (to keep shapes consistent across different
# variables). • Since a is sampled outside any plate, it does not gain plate
# dimensions and ends up as (6, 1, 1).
# 2. b.shape == (2, 1, 1, 1) • b ~ Bernoulli(p[a]) has 2 possible outcomes.
# Under enumeration, this adds a leftmost dimension of size 2. • Because b
# depends on a, Pyro still needs to keep the enumerated dimension for a in the
# shape, so overall there is an extra “(1, 1, 1)” to align with a’s enumerated
# dimension and potential downstream plates. • The net result is (2, 1, 1, 1).

# 3. c.shape == (2, 1, 1, 1, 1) • c ~ Bernoulli(0.3) is inside the plate of size
# 4 but is also enumerated (2 outcomes). • The leftmost dimension of size 2
# reflects these enumerated Bernoulli outcomes for c. • The extra “(1, 1, 1, 1)”
# reflects alignment with a, b, plus the unexpanded plate dimension. (At this
# point in the code, Pyro has not yet broadcast c fully to size 4; it has only
# inserted a placeholder dimension.)

# 4. d.shape == (2, 1, 1, 1, 1, 1) • d ~ Bernoulli(0.4) is also enumerated (2
# outcomes) and is within a nested plate of size 5 (d_plate). • As with c, we
# get an extra dimension of size 2 for enumeration, plus the existing “(1, 1, 1,
# 1, 1)” for alignment with a, b, c, and the plate structure. • Again, this is
# unexpanded across the 5 elements of d_plate at this stage. Hence (2, 1, 1, 1,
# 1, 1).

# 5. e.shape == (2, 1, 1, 1, 5, 4, 7) • e ~ Normal(e_loc, e_scale).to_event(1)
# is a continuous random variable (not enumerated), but it depends on d (so it
# must be expanded for each enumerated value of d). • Because e is inside both
# plates (the c_plate of size 4 and the d_plate of size 5), and e has event
# dimension 1 (size 7), the final shape factors in: – the enumerated dimension
# for d: size 2 (leftmost) – leftover broadcast dims from a, b, c: “(1, 1, 1)” –
# the plate dimensions for d_plate (size 5) and c_plate (size 4) – the event
# dimension 7 from Normal(…, …).to_event(1). • Thus (2, 1, 1, 1, 5, 4, 7).

# 6. e_loc.shape == (2, 1, 1, 1, 1, 1, 1) • e_loc = locs[d.long()].unsqueeze(-1)
# picks out either locs[0] or locs[1] depending on whether d = 0 or d = 1, so it
# has an enumerated dimension of size 2. • The trailing dimensions “(1, 1, 1, 1,
# 1, 1)” match up with the necessary broadcast for the plates and the
# to_event(1) dimension (which will later be expanded to size 7 by e_scale). •
# This is why e_loc is a rank-7 tensor of shape (2, 1, 1, 1, 1, 1, 1) at this
# stage.

# Putting this all together: • Each discrete sample (Categorical or Bernoulli)
# gains an extra enumeration dimension on the left (size 6 for a, size 2 for
# others). • Each plate used (of sizes 4 for c_plate and 5 for d_plate)
# typically appears to the right of enumeration dimensions. • The continuous
# variable e has an event dimension (size 7) that appears in the rightmost
# position because of .to_event(1). • Pyro internally broadcasts and reserves
# shape “slots” so that all enumerated/batch dimensions are consistent,
# resulting in those extra (1, 1, …) dummy dimensions.


@config_enumerate
def model3():
    p = pyro.param("p", torch.arange(6.0) / 6)
    locs = pyro.param("locs", torch.tensor([-1.0, 1.0]))

    a = pyro.sample("a", Categorical(torch.ones(6) / 6))
    b = pyro.sample("b", Bernoulli(p[a]))  # Note this depends on a.
    with pyro.plate("c_plate", 4):
        c = pyro.sample("c", Bernoulli(0.3))
        with pyro.plate("d_plate", 5):
            d = pyro.sample("d", Bernoulli(0.4))
            e_loc = locs[d.long()].unsqueeze(-1)
            e_scale = torch.arange(1.0, 8.0)
            e = pyro.sample("e", Normal(e_loc, e_scale).to_event(1))  # Note this depends on d.

    #                   enumerated|batch|event dims
    assert a.shape == (6, 1, 1)  # Six enumerated values of the Categorical.
    assert b.shape == (2, 1, 1, 1)  # Two enumerated Bernoullis, unexpanded.
    assert c.shape == (2, 1, 1, 1, 1)  # Only two Bernoullis, unexpanded.
    assert d.shape == (2, 1, 1, 1, 1, 1)  # Only two Bernoullis, unexpanded.
    assert e.shape == (2, 1, 1, 1, 5, 4, 7)  # This is sampled and depends on d.
    assert e_loc.shape == (
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert e_scale.shape == (7,)


test_model(model3, model3, TraceEnum_ELBO(max_plate_nesting=2))

trace = poutine.trace(poutine.enum(model3, first_available_dim=-3)).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())


p = pyro.param("p", torch.arange(6.0) / 6)
# locs = pyro.param("locs", torch.tensor([-1.0, 1.0]))

# a = pyro.sample("a", Categorical(torch.ones(6) / 6))
# b = pyro.sample("b", Bernoulli(p[a]))  # Note this depends on a.

print(p)
# print(a)
# print(p[a].item())
# print(b)
