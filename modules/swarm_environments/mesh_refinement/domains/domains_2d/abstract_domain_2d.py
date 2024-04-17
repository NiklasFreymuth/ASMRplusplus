from abc import ABC

import numpy as np

from modules.swarm_environments.mesh_refinement.domains import AbstractDomain


class AbstractDomain2D(AbstractDomain, ABC):
    """
    Abstract class for a 2D problem domain.
    Maintains a single member of a family of FEM problems.
    For a given domain, 2 meshes are maintained: The initial mesh and the integration mesh.
    - The initial mesh, which is the mesh that is initially given to the algorithm
    - The integration mesh, which is a fine-grained version of the mesh that can be used to calculate the precision of
        the learning algorithm using numerical integration
    """

    @property
    def boundary_line_segments(self) -> np.array:
        """
        The boundary of the domain, represented by line segements
        Returns: an array of shape (#line_segments, 4), where the last dimension is over (x0, y0, x1, y1)

        """
        boundary_edges = self._initial_mesh.boundary_facets()
        boundary_node_indices = self._initial_mesh.facets[:, boundary_edges]
        line_segments = self._initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        return line_segments
