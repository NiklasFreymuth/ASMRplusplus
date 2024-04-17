from typing import Any, Dict, Union

import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh_refinement.domains.domains_2d import AbstractDomain2D
from modules.swarm_environments.mesh_refinement.domains.domains_2d.extended_mesh_tri1 import (
    ExtendedMeshTri1,
)


class StarShape(AbstractDomain2D):
    """
    A quadrilateral geometry/domain. Specified by its 4 boundaries.
    Will always have its left boundary at x=0 and its right boundary at x=1
    """

    def __init__(
        self,
        domain_description_config: Dict[Union[str, int], Any],
        fixed_domain: bool,
        random_state: np.random.RandomState,
    ):
        """
        Initializes the domain of the pde to solve with the given finite element method
        Args:
            domain_description_config: Config containing additional details about the domain. Depends on the domain
            fixed_domain: Whether to use a fixed target domain. If True, a deterministic domain will be used.
            Else, a random one will be drawn
            random_state: The RandomState to use to draw the domain
        """
        self._center = np.array([0.5, 0.5])
        self._small_radius = 0.2
        self._large_radius = 0.5
        self._min_star_points = domain_description_config.get("min_star_points")
        self._max_star_points = domain_description_config.get("max_star_points")
        self._boundary_nodes = None
        self.maximum_position_distortion = domain_description_config.get("maximum_position_distortion")
        super().__init__(
            domain_description_config=domain_description_config,
            fixed_domain=fixed_domain,
            random_state=random_state,
        )

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Create an initial mesh

        Returns: The initial mesh on the sampled domain

        """
        if self.fixed_domain:
            star_points = (self._min_star_points + self._max_star_points) // 2
            num_points = 2 * star_points
            noise = np.zeros((2 * star_points, 2))
            rotation = 0
        else:
            star_points = self._random_state.randint(low=self._min_star_points, high=self._max_star_points + 1)
            num_points = 2 * star_points
            noise = self._random_state.uniform(
                -self.maximum_position_distortion,
                self.maximum_position_distortion,
                (num_points, 2),
            )
            rotation = self._random_state.uniform(0, 2 * np.pi)

        boundary_nodes = []
        for i in range(num_points):
            if i % 2 == 0:
                radius = self._small_radius
            else:
                radius = self._large_radius
            next_boundary_point = (
                self._center[0] + radius * np.cos(rotation + 2 * np.pi * i / num_points) + noise[i, 0],
                self._center[1] + radius * np.sin(rotation + 2 * np.pi * i / num_points) + noise[i, 1],
            )
            boundary_nodes.append(next_boundary_point)

        # normalize points
        boundary_nodes = np.array(boundary_nodes)
        boundary_nodes = (boundary_nodes - np.min(boundary_nodes, axis=0)) / (
            np.max(boundary_nodes, axis=0) - np.min(boundary_nodes, axis=0)
        )

        self._boundary_nodes = boundary_nodes

        initial_mesh = ExtendedMeshTri1.init_polygon(
            boundary_nodes=boundary_nodes,
            max_element_volume=self.max_initial_element_volume,
            initial_meshing_method=self.initial_meshing_method,
        )
        return initial_mesh

    @property
    def boundary_nodes(self) -> np.ndarray:
        assert self._boundary_nodes is not None
        return self._boundary_nodes
