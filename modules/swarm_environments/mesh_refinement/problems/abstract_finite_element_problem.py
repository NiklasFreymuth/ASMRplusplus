r"""
Base class for an Abstract (Static) Finite Element Problem.
The problem specifies a partial differential equation to be solved, and the boundary conditions. It also specifies the
domain/geometry of the problem.
Currently, uses a triangular mesh with linear elements.
"""
import abc
import copy
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from skfem import Basis, Mesh

from modules.swarm_environments.mesh_refinement.domains import create_domain
from modules.swarm_environments.mesh_refinement.problems.problem_util import ErrorIntegrator

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from plotly import graph_objects as go


class AbstractFiniteElementProblem(abc.ABC):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Initializes the AbstractFiniteElementProblem.

        Args:
            fem_config: Config containing additional details about the finite element method. Contains
                domain_config: Configuration describing the family of geometries for this problem
                error_metric:
                 The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean/average error is used
            random_state: A random state to use for drawing random numbers
        """
        self._random_state = random_state
        self._integration_error_metrics = ["squared", "mean", "maximum"]

        # initialize domain, pde on domain, and possibly error integrator
        self._domain = create_domain(
            domain_config=fem_config.get("domain"),
            random_state=copy.deepcopy(random_state),
        )
        self._set_pde()  # set pde after domain, since the domain may be used to generate the pde.
        self._set_error_integrator(fem_config=fem_config)
        self._assert_consistency()

    def _assert_consistency(self) -> None:
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _assert_consistency()")

    def _set_error_integrator(self, fem_config: Dict[Union[str, int], Any]) -> None:
        if fem_config.get("error_metric") in self._integration_error_metrics:  # do numerical integration
            self._error_integrator = ErrorIntegrator(
                scale_integration_by_area=fem_config.get("scale_integration_by_area"),
                error_metrics=self._integration_error_metrics,
                integration_mesh=self._domain.get_integration_mesh(),
                solution_calculation_function=self.calculate_solution,
                boundary_and_basis_creation_function=self.mesh_to_basis,
            )
        else:
            self._error_integrator = None

    def _set_pde(self) -> None:
        """
        Initializes a random PDE. This is not a reset, but a random initialization of the PDE.
        Returns:

        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _set_pde()")

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        By default, uses a linear triangular basis and no boundary on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement mesh_to_basis()")

    def calculate_solution(self, basis: Basis, cache: bool = False) -> np.array:
        """
        Wrapper for _calculate_solution that makes sure that the solution is a numpy array of
        shape (num_vertices, solution_dimension).
        Args:
            basis: The basis to calculate the solution for
            cache: Whether to cache information about the solution

        Returns: The solution for the given basis

        """
        solution = self._calculate_solution(basis=basis, cache=cache)
        if solution.ndim == 1:  # add a dimension if the solution is one-dimensional for consistency with interface
            solution = solution[:, None]
        return solution

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution of the underlying PDE for the given finite element basis.
        The solution is calculated for every *node/vertex* of the underlying mesh. The way it is calculated depends on
        the element used in the basis. Here, we use, ElementTriP1() elements, which are linear elements that will draw
        3 quadrature points for each element. These points lie in the middle of the edge between the barycenter of the
        element and its spanning nodes.
        The evaluation of the element is linearly interpolated based on those elements.

        Args:
            basis: The basis to calculate the solution for
            cache: Whether to cache information about the solution

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.


        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _calculate_solution")

    @property
    def has_different_nodal_solution(self) -> bool:
        """
        Whether the problem has a different nodal solution than the solution at the quadrature points.
        Usually, this corresponds to a higher-order basis or somesuch
        Returns:

        """
        return False

    @property
    def nodal_solution(self) -> np.array:
        raise NotImplementedError("AbstractFiniteElementProblem does not implement nodal_solution")

    def get_error_estimate_per_element(
        self, error_metric: str, basis: Basis, solution: np.array
    ) -> Dict[str, np.array]:
        """
        Calculates an error per *element/face* of the mesh in the given scikit-FEM basis.
        The error may be a scalar or a vector, depending on the problem.
        This error is calculated via integration over the domain and comparison to a fine-grained ground truth mesh.
        Args:
            basis: The basis to calculate the error estimate for
            solution: The solution of the pde in the given basis
            error_metric:
                The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean/average error is used
                    "maximum": the maximum error is used
        Returns:
            The error estimate per element/face of the mesh. This error estimate is an entity per element, and is
            calculated by integrating the error over the element.

        """
        if error_metric in self._integration_error_metrics:  # do numerical integration
            error_estimation_dict = self._error_integrator.get_error_estimate(
                coarse_basis=basis, coarse_solution=solution
            )
        else:
            error_estimation_dict = {}

        return error_estimation_dict

    #############################
    #        Observations       #
    #############################

    def element_features(self, mesh: Mesh, element_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_elements, num_features) containing the features for each element.
        Args:
            mesh: The mesh to use for the feature calculation
            element_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        Returns: An array of shape (num_elements, num_features) containing the features for each element.

        """
        raise NotImplementedError


    @staticmethod
    @abc.abstractmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        raise NotImplementedError("The solution dimension names are not defined for AbstractFiniteElementProblem")

    def project_to_scalar(self, values: np.array) -> np.array:
        """
        Projects a value per node and solution dimension to a scalar value per node.
        Equivalent to np.einsum('ij, j -> i', self._solution, self.fem_problem.solution_dimension_weights
        if simple weights are used
        Args:
            values: A vector of shape ([num_vertices,] solution_dimension)

        Returns: A scalar value per vertex
        """
        return np.dot(values, self.solution_dimension_weights)

    @property
    def solution_dimension_weights(self) -> np.array:
        """
        Returns a list of weights for the solution dimensions. This is used to weight the solution dimensions
        when calculating the error.
        Returns: A list of weights for the solution dimensions

        """
        if len(self.solution_dimension_names()) == 1:
            return np.array([1.0])
        else:
            raise NotImplementedError(
                "The solution dimension weights for solutions with multiple dimensions"
                " are not defined for AbstractFiniteElementProblem"
            )

    ##############
    # properties #
    ##############

    @property
    def boundary_line_segments(self) -> np.array:
        """
        The boundary of the domain, represented by line segements
        Returns: an array of shape (#line_segments, 4), where the last dimension is over (x0, y0, x1, y1)

        """
        return self._domain.boundary_line_segments

    @property
    def initial_mesh(self) -> Mesh:
        """
        Returns the initial mesh of the domain.
        Returns:

        """
        return self._domain.initial_mesh

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_mesh(self, mesh: Mesh) -> Dict[str, go.Figure]:
        """
        Returns a dictionary of additional plots that are used for visualization.
        Args:
            mesh: The mesh to use for the additional plots
        Returns:

        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement 'additional_plots_from_mesh'")

    def _get_scalar_plot(self, mesh: Mesh, scalars: np.ndarray, title: str = "Value") -> go.Figure:
        """
        Plot the given scalar values on the current mesh.
        Args:
            mesh: The scikit FEM mesh to use for the plot and the wireframe
            scalars: A value for each vertex or element of the mesh
            title: The title of the plot

        Returns: A plotly figure

        """
