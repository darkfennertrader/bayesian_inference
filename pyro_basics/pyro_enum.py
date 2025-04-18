# pylint: disable=not-context-manager, disable=protected-access

from typing import Any, Callable
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate, SVI
from pyro.distributions import constraints
from pyro.distributions import Categorical, MultivariateNormal, Normal, Bernoulli  # type: ignore
from pyro.optim import Adam  # pylint: disable=no-name-in-module # type: ignore
from pyro.ops.indexing import Vindex
import pyro.poutine as poutine
from pyro.util import ignore_jit_warnings
from trace_util import print_trace_summary


# 1 eumerated binary variable
@config_enumerate
def model_0():
    # A learnable parameter p; let's just start it at 0.3
    p = pyro.param("p", torch.tensor(0.3))

    # A single Bernoulli random variable (discrete)
    x = pyro.sample("x", dist.Bernoulli(p))

    # Return the value for demonstration
    return x


# 1 eumerated variable with three states
@config_enumerate
def model_1():
    # A learnable 3-state categorical parameter p
    # We constrain it to be a valid probability simplex.
    p = pyro.param("p", torch.tensor([0.2, 0.3, 0.5]), constraint=constraints.simplex)

    # Sample the discrete latent variable x with 3 states
    x = pyro.sample("x", dist.Categorical(p))

    # A second learnable parameter p_y capturing the conditional distribution p(y|x)
    # p_y is shaped (3, 2) for demonstration, indicating 3 possible x values
    # and 2 possible y values in each row.
    p_y = pyro.param(
        "p_y",
        torch.tensor(
            [
                [0.1, 0.9],  # p(y=0|x=0), p(y=1|x=0)
                [0.4, 0.6],  # p(y=0|x=1), p(y=1|x=1)
                [0.8, 0.2],  # p(y=0|x=2), p(y=1|x=2)
            ]
        ),
        constraint=constraints.simplex,
    )

    # Sample y conditioned on x by indexing into p_y[x]
    y = pyro.sample("y", dist.Categorical(p_y[x]))

    return x, y


# 1 eumerated variable with three states and 1 plate
@config_enumerate
def model_2(num_data=5):
    # A learnable prior over x with 3 states
    p = pyro.param("p", torch.tensor([0.2, 0.3, 0.5]), constraint=constraints.simplex)

    # A learnable conditional distribution p(y|x) with 3 x-states and 2 y-states
    p_y = pyro.param(
        "p_y",
        torch.tensor(
            [
                [0.1, 0.9],  # p(y=0|x=0), p(y=1|x=0)
                [0.4, 0.6],  # p(y=0|x=1), p(y=1|x=1)
                [0.8, 0.2],  # p(y=0|x=2), p(y=1|x=2)
            ]
        ),
        constraint=constraints.simplex,
    )

    # Replicate the sampling for 'num_data' observations
    with pyro.plate("data_plate", num_data):
        # Sample x for each data point (discrete, enumerated)
        x = pyro.sample("x", dist.Categorical(p))
        # Given x, sample y for each data point
        y = pyro.sample("y", dist.Categorical(p_y[x]))

    return x, y


# 2 eumerated variables: 1 is binary and the other has three states
@config_enumerate
def model_3():
    # First discrete variable with 3 possible values
    p_x = pyro.param("p_x", torch.tensor([0.4, 0.5, 0.1]), constraint=constraints.simplex)
    x = pyro.sample("x", dist.Categorical(p_x))

    # Second discrete variable with 2 possible values
    p_y = pyro.param("p_y", torch.tensor([0.7, 0.3]), constraint=constraints.simplex)
    y = pyro.sample("y", dist.Categorical(p_y))
    return x, y


@config_enumerate
def model_4(num_data=5):
    # Define parameters for the categorical distributions.
    p_x = pyro.param("p_x", torch.tensor([0.4, 0.5, 0.1]), constraint=constraints.simplex)
    p_y = pyro.param("p_y", torch.tensor([0.7, 0.3]), constraint=constraints.simplex)

    # Plate of size 5.
    with pyro.plate("plate_5", num_data):
        # First enumerated discrete variable with 3 possible values.
        x = pyro.sample("x", dist.Categorical(p_x))
        # Second enumerated discrete variable with 2 possible values.
        y = pyro.sample("y", dist.Categorical(p_y))


#####################################################################################

# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |R
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive probs_*
# with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has narrow
# treewidth, therefore admitting efficient inference by message passing. Pyro's
# TraceEnum_ELBO will find an efficient message passing scheme if one exists.


@config_enumerate
def model_5(sequences, lengths, args, batch_size=None, include_prior=True):

    # Often, Pyro’s dynamic control flow, enumeration, and other features are
    # not fully compatible with TorchScript tracing. By asserting that "we're
    # not tracing right now," the code can rely on Python-level control flow
    # (like for loops, enumerations, etc.) without worrying about TorchScript
    # constraints. If the assertion fails, it indicates we are in a JIT tracing
    # mode (which is unsupported for this model) and can avoid cryptic errors
    # later.
    assert not torch._C._get_tracing_state()

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
    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample(
            "probs_x", dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).to_event(1)
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
            "probs_y", dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2)
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
            x = pyro.sample(f"x_{i}_{t}", dist.Categorical(probs_x[x]))
            # print(x)
            with tones_plate:
                pyro.sample(f"y_{i}_{t}", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=sequence[t])


#####################################################################################


# Next let's make our simple model faster in two ways: first we'll support
# vectorized minibatches of data, and second we'll support the PyTorch jit
# compiler.  To add batch support, we'll introduce a second plate "sequences"
# and randomly subsample data to size batch_size.  To add jit support we
# silence some warnings and try to avoid dynamic program structure.


# Note that this is the "HMM" model in reference [1] (with the difference that
# in [1] the probabilities probs_x and probs_y are not MAP-regularized with
# Dirichlet and Beta distributions for any of the models)
@config_enumerate
def model_6(sequences, lengths, args, batch_size=None, include_prior=True):
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
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
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample(f"x_{t}", dist.Categorical(probs_x[x]))
                with tones_plate:
                    pyro.sample(
                        f"y_{t}", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=sequences[batch, t]
                    )


##############################################################################


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

    # print_trace_summary(model_0, enumerated=True, detailed=False)

    # print_trace_summary(model_1, enumerated=True, detailed=False)

    # print_trace_summary(model_4, enumerated=True, detailed=False)

    print_trace_summary(
        model_6, enumerated=True, detailed=False, sequences=sequences, lengths=lengths, args=args
    )
