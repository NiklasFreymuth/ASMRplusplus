from typing import Any, Dict, Union

import numpy as np

from modules.swarm_environments.mesh_refinement.domains.abstract_domain import AbstractDomain


def create_domain(*, domain_config: Dict[Union[str, int], Any], random_state: np.random.RandomState) -> AbstractDomain:
    """
    Builds and returns a domain class.
    Args:
        domain_config: Config containing additional details about the domain. Depends on the domain. Contains
            fixed_domain: Whether to use a fixed target domain or sample a random one
            domain_type: The type of domain to use.

        random_state: The RandomState to use to draw the domain in the __init__() call

    Returns: Some domain class that inherits from AbstractDomain.

    """
    fixed_domain = domain_config.get("fixed_domain")
    domain_type = domain_config.get("domain_type").lower()
    if domain_type == "convex_polygon":
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.convex_polygon import (
            ConvexPolygon,
        )

        domain = ConvexPolygon
    elif domain_type in ["star_shape", "star"]:
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.star_shape import (
            StarShape,
        )

        domain = StarShape
    elif domain_type == "trapezoid":
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.trapezoid import (
            Trapezoid,
        )

        domain = Trapezoid
    elif domain_type == "bottleneck":
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.bottleneck import (
            Bottleneck,
        )

        domain = Bottleneck
    elif domain_type == "trapezoid_hole":
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.trapezoid_hole import (
            TrapezoidHole,
        )

        domain = TrapezoidHole
    elif domain_type in ["square_hole", "symmetric_hole"]:
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.square_hole import (
            SquareHole,
        )

        domain = SquareHole
    elif domain_type in [
        "lshape",
        "lshaped",
        "l_shaped",
        "l_shape",
        "l-shaped",
        "l-shape",
    ]:
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.l_shape import LShape

        domain = LShape
    elif domain_type in ["big_square", "small_spiral", "large_spiral"]:
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.big_domain import (
            BigDomain,
        )

        domain = BigDomain
    elif domain_type in [
        "sq_symmetric",
        "square",
        "symmetric",
        "circle",
        "hexagon",
        "octagon",
    ]:
        from modules.swarm_environments.mesh_refinement.domains.domains_2d.simple_domain import (
            SimpleDomain,
        )

        domain = SimpleDomain

    # 3d domains!
    elif domain_type in ["cuboid", "cuboid3d"]:
        from modules.swarm_environments.mesh_refinement.domains.domains_3d.cuboid import Cuboid

        domain = Cuboid
    elif domain_type in ["truncated_cube", "truncated_cube3d"]:
        from modules.swarm_environments.mesh_refinement.domains.domains_3d.truncated_cube import (
            TruncatedCube,
        )

        domain = TruncatedCube
    else:
        raise ValueError(f"Unknown domain type '{domain_type}'")
    domain_instance = domain(
        domain_description_config=domain_config,
        fixed_domain=fixed_domain,
        random_state=random_state,
    )
    return domain_instance
