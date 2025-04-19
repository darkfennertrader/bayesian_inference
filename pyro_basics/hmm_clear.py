import os
import argparse
import logging
import sys
import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import (
    SVI,
    JitTraceEnum_ELBO,
    TraceEnum_ELBO,
    Trace_ELBO,
    config_enumerate,
    TraceTMC_ELBO,
)
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.optim import Adam  # pylint: disable=no-name-in-module # type: ignore
from pyro.util import ignore_jit_warnings

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 88 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive
# probs_* with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has
# narrow treewidth, therefore admitting efficient inference by message passing.
# Pyro's TraceEnum_ELBO will find an efficient message passing scheme if one
# exists.


def model_0(sequences, lengths, args, batch_size=None, include_prior=True):

    assert not torch._C._get_tracing_state()  # pylint: disable=protected-access
    num_sequences, max_length, data_dim = sequences.shape
    with poutine.mask(mask=include_prior):  # pylint: disable=not-context-manager
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).to_event(1),
        )
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2),
        )
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes. If
    # batch_size is None Pyro iterates over all sequences at once. To scale up
    # things we can set batch_size to a meaningful value (e.g. 16, 32, 64...)

    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    for i in pyro.plate("sequences", len(sequences), batch_size):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro.markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro.sample(...x[t-1]...).
            x = pyro.sample(
                f"x_{i}_{t}",
                dist.Categorical(probs_x[x]),
                infer={"enumerate": "parallel"},
            )
            with tones_plate:
                pyro.sample(
                    f"y_{i}_{t}",
                    dist.Bernoulli(probs_y[x.squeeze(-1)]),
                    obs=sequence[t],
                )


# Define args and parameters
class Args:
    hidden_dim = 16
    data_dim = 88
    max_length = 5
    batch_size = 1


args = Args()

# Generate fake data
num_sequences = 1
lengths = torch.tensor([5])
sequences = torch.bernoulli(torch.rand(num_sequences, args.max_length, args.data_dim))


# Dummy guide
def guide_0(sequences, lengths, args, batch_size=None, include_prior=True):
    pass


# Trace with enumeration
trace_enum = poutine.trace(poutine.enum(model_0, first_available_dim=-2)).get_trace(
    sequences, lengths, args, batch_size=args.batch_size, include_prior=True
)


# Function to skip Subsample nodes
def is_subsample(site):
    return type(site["fn"]).__name__ == "_Subsample"


# Correctly align and format output
print("## Sample Sites:")
print(f"{'site-name':>10} | {'dist shape':>12} | {'value shape':>12}")
print("-" * 42)

for name, site in trace_enum.nodes.items():
    if site["type"] == "sample" and not is_subsample(site):
        dist_shape = " ".join(map(str, site["fn"].batch_shape + site["fn"].event_shape))
        value_shape = " ".join(map(str, site["value"].shape))
        # printing nicely aligned output with fixed width
        print(f"{name:>10} | {dist_shape:>12} | {value_shape:>12}")
