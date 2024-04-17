from typing import Any, Dict, Union

import numpy as np
from skfem import MeshTet1

from modules.swarm_environments.mesh_refinement.domains.domains_3d import AbstractDomain3D
from modules.swarm_environments.mesh_refinement.domains.domains_3d.extended_mesh_tet1 import (
    ExtendedMeshTet1,
)


class TruncatedCube(AbstractDomain3D):
    """
    A class of triangular meshes for quadratic geometries with a quadratic hole in them.
    """

    def __init__(
        self,
        domain_description_config: Dict[Union[str, int], Any],
        fixed_domain: bool,
        random_state: np.random.RandomState,
    ):
        """
        Initializes the domain.
        Args:
            domain_description_config: Config containing additional details about the domain.
                Depends on the domain
            fixed_domain: Whether to use a fixed domain. If True, the same domain will be used
                throughout.
                If False, a family of geometries will be created and a new domain will be drawn
                from this family whenever the reset() method is called
            random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
                function class
        """
        self._mean_intersection_point = np.array([0.5, 0.5, 0.5])
        self.maximum_position_distortion = domain_description_config.get("maximum_position_distortion", 0.2)
        super().__init__(
            domain_description_config=domain_description_config,
            fixed_domain=fixed_domain,
            random_state=random_state,
        )

    def _get_initial_mesh(self) -> MeshTet1:
        """
        Reset the domain and create a new initial mesh.
        This method is only called once at the start of iff self.fixed_domain = False.

        A new domain is always drawn from a distribution specified by the config.

        This method is called by the environment when the reset() method is called.

        Returns: The boundary mesh of the new domain, i.e., the simplest mesh that describes the geometry of the domain

        """
        if self.fixed_domain:
            intersection_point = self._mean_intersection_point
        else:
            offset = self._random_state.uniform(
                low=-self.maximum_position_distortion,
                high=self.maximum_position_distortion,
                size=3,
            )
            intersection_point = self._mean_intersection_point + np.clip(offset, -0.3, 0.45)
        initial_mesh = ExtendedMeshTet1.init_truncated_cube(
            max_element_volume=self.max_initial_element_volume,
            intersection_point=intersection_point,
            initial_meshing_method=self.initial_meshing_method,
        )
        return initial_mesh
