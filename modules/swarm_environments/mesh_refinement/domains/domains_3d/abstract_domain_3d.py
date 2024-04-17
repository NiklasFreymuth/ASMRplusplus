from abc import ABC

from modules.swarm_environments.mesh_refinement.domains import AbstractDomain


class AbstractDomain3D(AbstractDomain, ABC):
    """
    Abstract class for a 3D problem domain.
    Maintains a single member of a family of FEM problems.
    For a given domain, 2 meshes are maintained: The initial mesh and the integration mesh.
    - The initial mesh, which is the mesh that is initially given to the algorithm
    - The integration mesh, which is a fine-grained version of the mesh that can be used to calculate the precision of
        the learning algorithm using numerical integration
    """
