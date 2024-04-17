r"""
Neumann Boundary Conditions on a 2D domain, using a Poisson equation.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from skfem import Basis, FacetBasis, LinearForm, Mesh, asm, condense, solve
from skfem.models.poisson import laplace
from util.function import wrapped_partial

from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    get_element_midpoints,
    get_line_segment_distances,
)
from modules.swarm_environments.mesh_refinement.problems.problems_2d.poisson import Poisson


def gaussian_inlet(x, mean, std):
    return np.exp(-((x - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))


def neumann(v, p, amplitude: float, frequency: int):
    """
    Returns the Neumann boundary conditions for the given basis and points. The Neumann boundary conditions are
    calculated based on the given basis and points. We assume a facet basis over a single line segment, which
    we map to the interval [0, 1] using a normal. We then query a  sine function combined with a parabolic
    function of the form (1-x)*x to get the Neumann boundary conditions.
    Args:
        v:
        p:
        amplitude: Scales the Neumann boundary conditions

    Returns:

    """
    # p.x is the positions of the nodes. there's also p.n for the normal vector and p.h for the element size
    positions = p.x
    example_normal = p.n[0, 0]  # all normals are the same as we consider a line segment

    a = positions.reshape(2, -1)
    projections = np.dot(a[:].T, example_normal)

    min_projection = np.min(projections)
    max_projection = np.max(projections)
    normalized_distances = (projections - min_projection) / (max_projection - min_projection)
    # shape ([x,y], num_points, 2), where the last dimension is over the integration points on the facet basis

    # apply the Gaussian function to the normalized distances, add some oscillation,
    # reshape back to integration points
    # neumann_boundary = gaussian_inlet(normalized_distances, mean, std)
    # neumann_boundary = neumann_boundary * np.sin(3 * np.pi * normalized_distances)
    neumann_boundary = (
        normalized_distances
        * (1 - normalized_distances)
        * np.sin(frequency * np.pi * normalized_distances)
        * 10
        * amplitude
    )
    neumann_boundary = neumann_boundary.reshape(-1, 2)

    return neumann_boundary * v


class NeumannPoisson(Poisson):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Extends the Poisson problem to include Neumann boundary conditions.
        """
        self._neumann_line_segments = None

        super().__init__(fem_config=fem_config, random_state=random_state)  # also calls reset()
        assert hasattr(self._domain, "boundary_nodes"), (
            "The domain must have a boundary_nodes attribute to use Neumann boundary conditions. "
            f"This is not the case for the domain {self._domain.__class__.__name__}"
        )

    def _set_pde(self) -> None:
        """
        Draw a new load function

        """
        self._set_neumann_boundary()
        super()._set_pde()  # resets load function

    def _set_neumann_boundary(self) -> None:
        """
        We assume that each boundary is between two boundary nodes, and that we consider a closed domain, i.e., that
        the first and last boundary node link up.
        Then, we can draw a random number of (node, node+1)-pairs and set Neumann boundaries on those

        Returns:

        """
        # create edges (node1, node2), (node2, node3), ..., (nodeN, node1)
        boundary_nodes = self._domain.boundary_nodes
        num_boundary_nodes = len(boundary_nodes)
        edges = np.array(
            [(boundary_nodes[i], boundary_nodes[(i + 1) % num_boundary_nodes]) for i in range(num_boundary_nodes)]
        )

        # assign an index to each edge. 0: no boundary (Neumann with 0 value), 1: Neumann, 2: zero Dirichlet
        edge_idxs = np.random.choice([0, 1, 2], size=len(edges), p=[0.0, 0.5, 0.5])
        # always set at least one edge to be Neumann, and one to be zero Dirichlet.
        # Then shuffle to make the order random
        edge_idxs[0] = 1
        edge_idxs[1] = 2
        edge_idxs = self._random_state.permutation(edge_idxs)

        # store boundary as # (x0, y0, x1, y1) line segments
        self._neumann_line_segments = edges[edge_idxs == 1].reshape(-1, 4)
        self._zero_dirichlet_line_segments = edges[edge_idxs == 2].reshape(-1, 4)

        # draw distributions for the Neumann boundary conditions
        num_neumann_segments = self._neumann_line_segments.shape[0]
        self._random_amplitudes = self._random_state.random(num_neumann_segments) * 2 + 1
        self._random_frequencies = self._random_state.randint(low=2, high=4, size=num_neumann_segments) * 2 - 1
        # self._random_frequencies = np.ones(num_neumann_segments) * 5

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution for the parameterized Poisson equation based on the given basis. The solution is
        calculated for every node/vertex of the underlying mesh, and the way it is calculated depends on the element
        used in the basis.
        For example, ElementTriP1() elements will draw 3 quadrature points for each face that lie in the middle of the
        edge between the barycenter of the face and its spanning nodes, and then linearly interpolate based on those
        elements.
        Args:
            cache:

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.

        """
        K = asm(laplace, basis)  # finite element assembly. Returns a sparse matrix
        f = asm(LinearForm(self.load), basis)
        # f is the rhs of the linear system that matches the load function
        # here, we add the Neumann boundary conditions to the rhs of the linear system, which is unique to the fem

        mesh = basis.mesh
        boundary_facets = mesh.facets[:, mesh.boundary_facets()]
        points = mesh.p[:, boundary_facets]
        points = points.T.reshape(-1, 2)

        f += self._add_neumann_boundaries(basis, points)
        D = self._get_dirichlet_dofs(basis, points)

        condensed_system = condense(K, f, D=D)
        solution = solve(*condensed_system)
        return solution

    def _add_neumann_boundaries(self, basis, points):
        """
        Adds Neumann boundary conditions to the rhs of the linear system that is solved to calculate the solution of the
        parameterized Poisson equation. The Neumann boundary conditions are added based on the given basis and points.

        Args:
            basis: The scikit fem basis to use for the calculation
            points: An array of points on the boundary of the mesh

        Returns: An array of shape (num_dofs, ) containing the Neumann boundary conditions for each degree of freedom.

        """

        # get distances to the neumann line segments
        neumann_distances = get_line_segment_distances(points=points, projection_segments=self._neumann_line_segments)
        # get elements that are on the boundary
        neumann_mask = np.isclose(neumann_distances, 0)
        # add Neumann boundary conditions
        num_neumann_segments = self._neumann_line_segments.shape[0]
        all_masks = neumann_mask.reshape(-1, 2, num_neumann_segments).all(axis=1)
        # shape (num_facets, num_neumann_segments).
        # The .all() selects facets for which all points are on the line segment
        neumann_rhs = 0
        for segment_idx in range(num_neumann_segments):
            # loop over all segments and add the Neumann boundary conditions for each segment
            segment_mask = all_masks[:, segment_idx]
            segment_facet_idxs = basis.mesh.boundary_facets()[segment_mask]
            segment_basis = FacetBasis(basis.mesh, basis.elem, facets=segment_facet_idxs)
            partial_neumann = wrapped_partial(
                neumann,
                amplitude=self._random_amplitudes[segment_idx],
                frequency=self._random_frequencies[segment_idx],
            )
            b = asm(LinearForm(partial_neumann), segment_basis)
            neumann_rhs += b  # can simply add the Neumann boundary conditions to the rhs in the fem
        return neumann_rhs

    def _get_dirichlet_dofs(self, basis: Basis, points: np.ndarray):
        """
        Returns the Dirichlet degrees of freedom for the given basis and points. The Dirichlet degrees of freedom are
        the degrees of freedom that are fixed to a certain value (here: 0), and are not part of the linear system that
        is solved. We choose only some of the boundary segments to be zero Dirichlet, and leave the rest as Neumann
        boundary conditions or unconstrained for a more interesting problem.
        Args:
            basis: The scikit fem basis to use for the calculation
            points: An array of points on the boundary of the mesh

        Returns: An array of shape (num_dirichlet_dofs, ) containing the indices of the Dirichlet degrees of freedom.

        """
        num_dirichlet_segments = self._zero_dirichlet_line_segments.shape[0]
        dirichlet_distances = get_line_segment_distances(
            points=points, projection_segments=self._zero_dirichlet_line_segments
        )
        dirichlet_mask = np.isclose(dirichlet_distances, 0)
        dirichlet_mask = dirichlet_mask.reshape(-1, 2, num_dirichlet_segments).all(axis=1).any(axis=1)
        dirichlet_idxs = basis.mesh.boundary_facets()[dirichlet_mask]
        D = basis.get_dofs(dirichlet_idxs)
        return D

    ##############################
    #         Observations       #
    ##############################
    def element_features(self, mesh: Mesh, element_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_elements, num_features) containing the features for each element.
        Args:
            mesh: The mesh to use for the feature calculation
            element_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        Returns: An array of shape (num_elements, num_features) containing the features for each element.
        """
        features = []

        if "distance_to_neumann" in element_feature_names:
            distance_to_inlet = get_line_segment_distances(
                points=get_element_midpoints(mesh),
                projection_segments=self._neumann_line_segments,
                return_minimum=True,
            )
            features.append(distance_to_inlet)
        if "distance_to_dirichlet" in element_feature_names:
            distance_to_dirichlet = get_line_segment_distances(
                points=get_element_midpoints(mesh),
                projection_segments=self._zero_dirichlet_line_segments,
                return_minimum=True,
            )
            features.append(distance_to_dirichlet)

        features = np.array(features).T
        poisson_features = super().element_features(mesh=mesh, element_feature_names=element_feature_names)
        features = np.concatenate((features, poisson_features), axis=1)

        return features
