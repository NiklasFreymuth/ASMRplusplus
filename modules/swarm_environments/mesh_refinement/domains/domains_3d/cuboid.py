from typing import Any, Dict, Union

import numpy as np
from skfem import MeshTet1

from modules.swarm_environments.mesh_refinement.domains.domains_3d import AbstractDomain3D
from modules.swarm_environments.mesh_refinement.domains.domains_3d.extended_mesh_tet1 import (
    ExtendedMeshTet1,
)


class Cuboid(AbstractDomain3D):
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
            fixed_domain: Whether to use a fixed domain. Since the cube is a simple geometry, we do not support
                random domains.
            random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
                function class
        """
        from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
            format_3d_boundary,
        )

        self._boundary = format_3d_boundary(domain_description_config.get("boundary"))
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
        initial_mesh = ExtendedMeshTet1.init_cuboid(
            max_element_volume=self.max_initial_element_volume,
            initial_meshing_method=self.initial_meshing_method,
            lengths=self._boundary[3:],
        )
        return initial_mesh
