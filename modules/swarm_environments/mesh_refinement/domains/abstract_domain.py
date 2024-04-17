import abc
import copy
from typing import Any, Dict, Union

import numpy as np
from skfem import MeshTri1


class AbstractDomain(abc.ABC):
    """
    Abstract class for a problem domain.
    Maintains a single member of a family of FEM problems.
    For a given domain, 2 meshes are maintained: The initial mesh and the integration mesh.
    - The initial mesh, which is the mesh that is initially given to the algorithm
    - The integration mesh, which is a fine-grained version of the mesh that can be used to calculate the precision of
        the learning algorithm using numerical integration
    """

    def __init__(
        self,
        *,
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
        self.max_initial_element_volume: float = domain_description_config.get("max_initial_element_volume")
        self.num_integration_refinements: int = domain_description_config.get("num_integration_refinements")
        self.initial_meshing_method: str = domain_description_config.get("initial_meshing_method").lower()
        self.fixed_domain = fixed_domain
        self._random_state = random_state
        self._initial_mesh: MeshTri1 = self._get_initial_mesh()

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Internal function for the actual resetting. This function is called by the reset() method
        if there is no boundary mesh or if not self.fixed_domain, i.e., if a new domain should be drawn from the family

        Returns: The boundary mesh of the new domain, i.e., the simplest mesh that describes the geometry of the domain

        """
        raise NotImplementedError("AbstractDomain does not implement '_get_initial_mesh()'")

    @property
    def initial_mesh(self) -> MeshTri1:
        """
        Returns: The initial mesh for the algorithm. Maintains a cache for this mesh, so that it is only created once
            or after the domain has been reset.
            Returns a copy of the mesh to avoid accidental changes to the cached version by the algorithm
        """
        return copy.deepcopy(self._initial_mesh)

    def get_integration_mesh(self) -> MeshTri1:
        """
        Returns: The integration mesh, which is a fine-grained mesh that can be compared against the algorithm mesh
            to calculate the precision/accuracy of the mesh created by the algorithm.
        """
        return self._initial_mesh.refined(int(self.num_integration_refinements))
