r"""Laplacian task with an outer boundary with boundary condition Omega_0=0,
and an inner boundary with condition Omega_1=1
"""

import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import skfem as fem
from plotly import graph_objects as go
from skfem import Basis, ElementTriP1, Mesh
from skfem.models.poisson import laplace

from modules.swarm_environments.mesh_refinement.domains.domains_2d.square_hole import SquareHole
from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    get_element_midpoints,
    get_line_segment_distances,
)
from modules.swarm_environments.mesh_refinement.problems.problems_2d.abstract_finite_element_problem_2d import (
    AbstractFiniteElementProblem2D,
)

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def in_rectangle(x: np.array, rectangle: np.array) -> np.array:
    """
    Args:
        x: Points to query. Must have shape (2,...)
        rectangle: 4-tuple (x1, y1, x2, y2) that describes the rectangle boundaries
    Returns:
        A boolean array of shape (...)
    """
    return np.logical_and(
        np.logical_and(x[0] >= rectangle[0], x[0] <= rectangle[2]),
        np.logical_and(x[1] >= rectangle[1], x[1] <= rectangle[3]),
    )


class Laplace(AbstractFiniteElementProblem2D):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Args:
            fem_config: A dictionary containing the configuration for the finite element method.
            random_state: A random state to use for reproducibility.

        """
        self._temperature = 1.0  # fixed material parameter

        super().__init__(fem_config=fem_config, random_state=random_state)  # also calls reset() and thus _set_pde()
        assert isinstance(self._domain, SquareHole), (
            f"Laplace task currently only defined " f"for SquareHole domain, given '{type(self._domain)}'"
        )

    def _set_pde(self) -> None:
        """
        This function is called to draw a new PDE from a family of available PDEs.
        Since the PDE does not change for this task, we do not do anything here except to grab the current hole boundary

        """
        self._source_rectangle = self._domain.hole_boundary

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        By default, uses a linear triangular basis and no boundary on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        source_facets = mesh.facets_satisfying(
            partial(in_rectangle, rectangle=self._source_rectangle),
            boundaries_only=True,
        )
        mesh_ = mesh.with_boundaries(
            {
                "source": source_facets,
            }
        )

        basis = Basis(mesh_, ElementTriP1())
        return basis

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

        Returns: An array (num_vertices, 2), where every entry corresponds to a vector of the norm of the stress
        at the corresponding vertex, and the magnitude of the displacement at the corresponding vertex.

        """

        # set boundary conditions
        # Get all degrees of freedom and set appropriate entry to prescribed BCs.
        boundary_temperature = basis.zeros()
        boundary_temperature[basis.get_dofs({"source"})] = self._temperature

        # Assemble matrices, solve problem
        matrix = fem.asm(laplace, basis)
        solution = fem.solve(*fem.condense(matrix, x=boundary_temperature, I=basis.mesh.interior_nodes()))
        return solution

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
        if "distance_to_source" in element_feature_names:
            return np.array(
                [
                    get_line_segment_distances(
                        points=get_element_midpoints(mesh),
                        projection_segments=self.source_line_segments,
                        return_minimum=True,
                    )
                ]
            ).T
        else:
            return None


    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["temperature"]

    @property
    def source_line_segments(self):
        source_facets = self.initial_mesh.facets_satisfying(
            partial(in_rectangle, rectangle=self._source_rectangle),
            boundaries_only=True,
        )
        boundary_node_indices = self.initial_mesh.facets[:, source_facets]
        line_segments = self.initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        return line_segments

    ###############################
    # plotting utility functions #
    ###############################

    def _minimum_source_distance_plot(self, mesh: Mesh) -> go.Figure:
        """
        Plots the minimum distance to the heat source for each face of the mesh
        Args:
            mesh:

        Returns:

        """
        element_midpoints = mesh.p[:, mesh.t].mean(axis=1).T
        face_distances = get_line_segment_distances(
            points=element_midpoints,
            projection_segments=self.source_line_segments,
            return_minimum=True,
        )
        return self._get_scalar_plot(mesh=mesh, scalars=face_distances, title="Min. Heat Source Distance")

    def additional_plots_from_mesh(self, mesh) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            mesh:

        """
        additional_plots = {"mininum_source_distance": self._minimum_source_distance_plot(mesh=mesh)}
        return additional_plots
