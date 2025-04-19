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


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 88 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive probs_*
# with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has narrow
# treewidth, therefore admitting efficient inference by message passing. Pyro's
# TraceEnum_ELBO will find an efficient message passing scheme if one exists.


def model_0(sequences, lengths, args, batch_size=None, include_prior=True):

    # Often, Pyro’s dynamic control flow, enumeration, and other features are
    # not fully compatible with TorchScript tracing. By asserting that "we're
    # not tracing right now," the code can rely on Python-level control flow
    # (like for loops, enumerations, etc.) without worrying about TorchScript
    # constraints. If the assertion fails, it indicates we are in a JIT tracing
    # mode (which is unsupported for this model) and can avoid cryptic errors
    # later.
    assert not torch._C._get_tracing_state()  # pylint: disable=protected-access

    # if sequences has shape (N, T, D), then N = num_sequences (batch
    # dimension), T = max_length (# of timesteps),
    # and D = data_dim (dimension of the observation y - 88 piano keys).
    num_sequences, max_length, data_dim = sequences.shape

    # this context manager allows to decide whether or not include the priors in
    # the inference. When include_prior = True, you include those global terms
    # in the loss function. That means your inference will properly balance the
    # data likelihood with your prior’s constraints. When include_prior = False,
    # you zero out the contribution of those global terms so that they do not
    # affect training. args.hidden_dim determines the number of possible
    # discrete hidden states. In other words, x is a categorical variable that
    # can take one of args.hidden_dim possible values.
    with poutine.mask(mask=include_prior):  # pylint: disable=not-context-manager
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).to_event(1),
        )

        # Because alpha=0.1 and beta=0.9 sum to 1.0, the prior’s mean is
        # 0.1/(0.1+0.9) = 0.1 (10%). Having alpha+beta=1.0 is relatively small,
        # so the Beta distribution is quite flexible and not tightly peaked.
        # This allows the posterior to move away from 0.1 to wherever the data
        # suggests with little “resistance.” The intuition matches the rough
        # idea that around 4 of 88 notes (roughly 4.5%) might be active, so
        # setting the mean near 10% is a small structural bias that still leaves
        # room for variation once you observe actual data.
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2),
        )
    # print("\ninside model lengths:")
    # print(lengths)
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    for i in pyro.plate("sequences", len(sequences), batch_size):
        print(f"\nLOOP {i}:")
        # determines the lenght of the specific sequence in the minibatch
        length = lengths[i]
        # extract the specific sequence
        sequence = sequences[i, :length]
        # print(sequence)
        x = 0
        for t in pyro.markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro.sample(...x[t-1]...).
            # print(probs_x[x])
            x = pyro.sample(
                f"x_{i}_{t}",
                dist.Categorical(probs_x[x]),
                infer={"enumerate": "parallel"},
            )
            # print(x)
            with tones_plate:
                pyro.sample(
                    f"y_{i}_{t}",
                    dist.Bernoulli(probs_y[x.squeeze(-1)]),
                    obs=sequence[t],
                )


#################################################################################

# Next let's make our simple model faster in two ways: first we'll support
# vectorized minibatches of data, and second we'll support the PyTorch jit
# compiler.  To add batch support, we'll introduce a second plate "sequences"
# and randomly subsample data to size batch_size.  To add jit support we silence
# some warnings and try to avoid dynamic program structure.


# Note that this is the "HMM" model in reference [1] (with the difference that
# in [1] the probabilities probs_x and probs_y are not MAP-regularized with
# Dirichlet and Beta distributions for any of the models)
def model_1(sequences, lengths, args, batch_size=None, include_prior=True):
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length

    with poutine.mask(mask=include_prior):  # pylint: disable=not-context-manager
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).to_event(1),
        )
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2),
        )
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro.plate("sequences", num_sequences, subsample_size=batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(  # pylint: disable=not-context-manager
                mask=(t < lengths).unsqueeze(-1)
            ):
                x = pyro.sample(
                    f"x_{t}",
                    dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"},
                )
                with tones_plate:
                    pyro.sample(
                        f"y_{t}",
                        dist.Bernoulli(probs_y[x.squeeze(-1)]),
                        obs=sequences[batch, t],
                    )


def guide_0(sequences, lengths, args, batch_size=None, include_prior=True):
    pass


models = {
    name[len("model_") :]: model for name, model in globals().items() if name.startswith("model_")
}


def trace_shapes(model, sequences, lengths, args):
    # First, generate a guide trace to replay enumeration dimensions correctly
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))  # type: ignore

    max_plate_nesting = 1 if model is model_0 else 2
    first_available_dim = -2 if model is model_0 else -3

    # Run a dummy step to setup parameters in Pyro’s paramstore
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, loss=TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))

    svi.step(sequences, lengths, args, batch_size=args.batch_size)

    guide_trace = poutine.trace(guide).get_trace(
        sequences, lengths, args, batch_size=args.batch_size
    )

    # Enumerate the model with replay
    model_trace = poutine.trace(
        poutine.replay(poutine.enum(model, first_available_dim=first_available_dim), guide_trace)
    ).get_trace(sequences, lengths, args=args, batch_size=args.batch_size)

    records = []
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":  # type: ignore
            if isinstance(site["fn"], torch.distributions.Distribution):  # type: ignore
                dist_shape = " ".join(map(str, site["fn"].batch_shape + site["fn"].event_shape))  # type: ignore
                val_shape = " ".join(map(str, site["value"].shape))  # type: ignore
            else:  # plates
                dist_shape = ""
                val_shape = " ".join(map(str, site["value"].shape))  # type: ignore

            records.append({"Sample Site": name, "dist": dist_shape, "value": val_shape})

    df = pd.DataFrame(records)
    df.set_index("Sample Site", inplace=True)
    pd.set_option("display.max_colwidth", None)
    print()
    print(df)


if __name__ == "__main__":

    # Define args and parameters
    class Args:
        num_sequences = 10
        hidden_dim = 16  # 16
        data_dim = 88  # 88 dimension of independent tones y[t] given the hidden variable x[t]
        max_length = 5  # time steps
        batch_size = 10
        jit = False

    args = Args()

    # Generate fake data
    lengths = torch.tensor([args.max_length] * args.num_sequences)
    pyro.set_rng_seed(42)
    sequences = torch.bernoulli(torch.rand(args.num_sequences, args.max_length, args.data_dim))
    print(sequences.shape)
    print("\nOverall Sequences:")
    # print(sequences)
    print(sequences.shape)

    # Call trace_shapes function
    trace_shapes(model_1, sequences, lengths, args)
