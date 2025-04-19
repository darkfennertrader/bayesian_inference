import os
import argparse
import logging
import sys
import pandas as pd
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

# Add another handler for logging debugging events (e.g. for profiling) in a
# separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)

# Below is a tiny toy example with two Hidden Markov Models (HMMs) in Pyro,
# illustrating the difference between a simple/naive version (model_0) and a
# more batch-friendly vectorized version (model_1). To keep it very simple,
# consider a 2-state HMM that models coin flips:

# • Each latent state x[t] ∈ {0,1} tells us which of two coins is being used at
# time t. • We draw each state x[t] from a transition distribution p(x[t] |
# x[t-1]). • Then we observe a coin flip outcome y[t] ∈ {0,1} from the Bernoulli
# emission distribution associated with x[t].

# In model_0, we loop over sequences one at a time (slower for large datasets).
# In model_1, we vectorize over the sequence plate (faster, because it uses
# batched tensor operations).


def model_0(data, lengths, hidden_dim=2):
    """
    A naive HMM model that loops over each sequence one by one.

    Arguments:
    data: a float tensor of shape [num_sequences, max_length],
            with 0/1 coin flip outcomes
    lengths: an int tensor of shape [num_sequences], specifying
            the actual length of each sequence
    hidden_dim: number of hidden states (2, in this toy example)
    """
    num_sequences, max_length = data.shape

    # 1. Sample a 2x2 transition matrix for x[t]
    #    Using a Dirichlet so that each row of trans_x is a category distribution
    trans_x = pyro.sample("trans_x", dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1))

    # 2. Sample emissions (coin flip probabilities) for each state
    #    Using Beta(0.1, 0.9) as a weak prior that roughly centers near 0.1
    emit_probs = pyro.sample("emit_probs", dist.Beta(0.1, 0.9).expand([hidden_dim]).to_event(1))

    # 3. Loop over sequences one at a time.
    for i in pyro.plate("sequences", num_sequences):
        T = lengths[i]
        # Initialize x to 0 for the first time step.
        # (In a real model, you might sample x[0] from a prior or param.)
        x_prev = 0
        for t in pyro.markov(range(T)):
            # Sample the hidden state x[t], depends on x[t-1]
            x_t = pyro.sample(
                f"x_{i}_{t}",
                dist.Categorical(trans_x[x_prev]),
                infer={"enumerate": "parallel"},
            )
            # Observe the coin flip y[t] given x[t]
            pyro.sample(
                f"y_{i}_{t}",
                dist.Bernoulli(emit_probs[x_t]),
                obs=data[i, t],
            )
            x_prev = x_t


def model_1(data, lengths, hidden_dim=2, batch_size=None):
    """
    A faster HMM model that vectorizes over sequences.

    Arguments:
    data: a float tensor of shape [num_sequences, max_length]
    lengths: an int tensor of shape [num_sequences]
    hidden_dim: number of hidden states
    batch_size: optional subsample size
    """
    num_sequences, max_length = data.shape

    # 1. Draw transition probabilities
    trans_x = pyro.sample("trans_x", dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1))

    # 2. Draw emission probabilities
    emit_probs = pyro.sample("emit_probs", dist.Beta(0.1, 0.9).expand([hidden_dim]).to_event(1))

    # 3. Subsample and vectorize over sequences
    # Notice we use pyro.plate(..., dim=-2) for sequences
    with pyro.plate("sequences", num_sequences, subsample_size=batch_size, dim=-2) as batch:
        # Extract just the batch of data
        data_batch = data[batch]
        lengths_batch = lengths[batch]

        # Initialize x=0 for all sequences in the batch
        x_prev = torch.zeros(data_batch.shape[0], dtype=torch.long)

        # 4. Loop over time steps. We can do so up to max_length, but we apply masks
        #    so that steps beyond each length are ignored.
        for t in pyro.markov(range(max_length)):
            mask_t = t < lengths_batch  # True if t < sequence length
            # Sample x[t] for the entire batch in one call
            x_t = pyro.sample(
                f"x_{t}",
                dist.Categorical(trans_x[x_prev]),
                infer={"enumerate": "parallel"},
            )

            # Observe the coin flips from Bernoulli(emit_probs[x_t])
            pyro.sample(
                f"y_{t}",
                dist.Bernoulli(emit_probs[x_t]),
                obs=data_batch[:, t],
                # Only include these observations for sequences where t < length
            ).mask(  # type: ignore
                mask_t
            )

            # Update x_prev for the next time step
            x_prev = x_t * mask_t  # multiply by mask_t to keep state 0 if masked out


if __name__ == "__main__":
    pass
