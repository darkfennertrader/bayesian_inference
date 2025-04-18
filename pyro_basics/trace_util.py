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


def infer_enum_dimensions(model, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}

    enum_dims_used = []
    plate_dims_used = []

    def config_fn(msg):
        infer_config = msg.get("infer", {})
        enum_dims = infer_config.get("enum_discrete_used_dims", None)
        if enum_dims is not None:
            enum_dims_used.extend(enum_dims)

        cond_indep_stack = msg.get("cond_indep_stack", ())
        for frame in cond_indep_stack:
            if frame.vectorized:
                plate_dims_used.append(frame.dim)

        return infer_config

    wrapped_model = poutine.infer_config(fn=model, config_fn=config_fn)
    trace = poutine.trace(wrapped_model).get_trace(*args, **kwargs)

    # If no plates are used, place enumeration at dim = -1 (rather than -2).
    if plate_dims_used:
        leftmost_plate_dim = min(plate_dims_used)
        first_available_dim = leftmost_plate_dim - 1
    else:
        first_available_dim = -1

    return first_available_dim


def print_trace_summary(
    model: Callable[..., Any],
    *model_args: Any,
    enumerated: bool = False,
    detailed: bool = False,
    **model_kwargs: Any,
) -> None:
    """
    Runs model once to produce a trace, then prints shape information.

    Parameters:
    -----------
    model : callable
        A Pyro model (possibly decorated with @config_enumerate).
    enumerated : bool
        If True, wrap the model with `poutine.enum(model)`, which enumerates
        any eligible discrete variables.
    detailed : bool
        If False, prints a concise summary (trace.format_shapes()).
        If True, prints a detailed site-by-site summary, including the
        distribution object, value shape, and log_prob shape.

    *model_args, **model_kwargs : arguments
        Any arguments your model needs.
    """

    # Optionally wrap the model for enumeration
    if enumerated:
        first_avail_dim = infer_enum_dimensions(model, model_args, model_kwargs)
        # first_avail_dim = -1
        print(first_avail_dim)

        traced_model = poutine.enum(model, first_available_dim=first_avail_dim)
    else:
        traced_model = model

    # Generate a trace
    trace = poutine.trace(traced_model).get_trace(*model_args, **model_kwargs)
    # Ensure that log probabilities are computed (so log_prob shows up)
    trace.compute_log_prob()

    if not detailed:
        # Concise summary
        print(trace.format_shapes())
    else:
        print("DETAILED TRACE INSPECTION:")
        for site_name, node in trace.nodes.items():
            # Safely fetch the type of this node
            site_type = node.get("type", None)
            if site_type == "sample":
                print(f"Sample site: {site_name}")
                # Distribution object, if present
                distribution_obj = node.get("fn", None)
                if distribution_obj is not None:
                    print(f"  distribution object: {distribution_obj}")

                # 'value' is where the sampled or enumerated data is stored
                value_tensor = node.get("value", None)
                if value_tensor is not None:
                    print(f"  value shape: {tuple(value_tensor.shape)}")

                # 'log_prob' is present only if we called compute_log_prob()
                log_prob_tensor = node.get("log_prob", None)
                if log_prob_tensor is not None:
                    print(f"  log_prob shape: {tuple(log_prob_tensor.shape)}")

                print()

            elif site_type == "param":
                print(f"Param site: {site_name}")
                # The parameter's actual tensor
                param_tensor = node.get("value", None)
                if param_tensor is not None:
                    print(f"  param shape: {tuple(param_tensor.shape)}")
                print()

            elif site_type == "plate":
                print(f"Plate site: {site_name}")
                # Plate size is typically stored in node['args'] if it was specified
                plate_args = node.get("args", ())
                if plate_args:
                    plate_size = plate_args[0]
                    print(f"  plate size: {plate_size}")
                cond_indep_stack = node.get("cond_indep_stack", None)
                if cond_indep_stack is not None:
                    print(f"  cond_indep_stack: {cond_indep_stack}")
                print()
            # For completeness, you could also handle other node types
            # 'control_flow', or custom names.


if __name__ == "__main__":
    pass
