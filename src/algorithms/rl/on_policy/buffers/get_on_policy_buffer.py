from typing import Optional

import torch

from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import (
    AbstractMultiAgentOnPolicyBuffer,
)


def get_on_policy_buffer(
        buffer_size: int,
        gae_lambda: float,
        discount_factor: float,
        value_function_aggr: str,
        mixed_return_config: Optional[dict] = None,
        projection_type: Optional[str] = None,
        device: Optional[torch.device] = None,
        normalize_mappings: bool = False,
) -> AbstractMultiAgentOnPolicyBuffer:
    """

    Args:
        buffer_size:
        gae_lambda:
        discount_factor:
        value_function_aggr: The scope of the value function.
        Either "agent" for a value per agent/node or "mean"/"sum"/"max" for a single value for each graph/set of agents
        mixed_return_config:
        normalize_mappings:
            the agent mappings map multiple agents to one agent, so we need to normalize the total weight
            of the mappings by the ratio of agents of the current and previous step to preserve the
            "total mass" of quantities when calculating the advantages and returns

    Returns: A corresponding OnPolicyBuffer that can deal with either node- or graph-wise value functions

    """
    if value_function_aggr == "spatial":
        use_mixed_return = mixed_return_config.get("global_weight", 0) > 0
        if use_mixed_return:
            from src.algorithms.rl.on_policy.buffers.mixed_return_on_policy_buffer import (
                MixedOnPolicyBuffer,
            )

            return MixedOnPolicyBuffer(
                buffer_size=buffer_size,
                gae_lambda=gae_lambda,
                discount_factor=discount_factor,
                device=device,
                mixed_return_config=mixed_return_config,
                normalize_mappings=normalize_mappings,
                projection_type=projection_type,
            )
        else:
            from src.algorithms.rl.on_policy.buffers.spatial_on_policy_buffer import (
                SpatialOnPolicyBuffer,
            )

            return SpatialOnPolicyBuffer(
                buffer_size=buffer_size,
                gae_lambda=gae_lambda,
                discount_factor=discount_factor,
                device=device,
                normalize_mappings=normalize_mappings,
                projection_type=projection_type,
            )
    elif value_function_aggr in ["mean", "sum", "max"]:
        from src.algorithms.rl.on_policy.buffers.graph_on_policy_buffer import (
            GraphOnPolicyBuffer,
        )

        return GraphOnPolicyBuffer(
            buffer_size=buffer_size,
            gae_lambda=gae_lambda,
            discount_factor=discount_factor,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unknown value_function_aggr '{value_function_aggr}'")
