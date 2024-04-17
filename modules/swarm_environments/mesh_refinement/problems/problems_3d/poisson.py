r"""
3d Poisson equation.
The poisson equation is given as \Delta u = f, where \Delta is the Laplacian, u is the solution, and f is the
load.
We consider 3D cuboid domains with zero boundary conditions along x and y, and no boundary conditions along z.
"""
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from skfem import Basis, ElementTetP1, LinearForm, Mesh, asm, condense
from skfem.models import laplace
from util.function import wrapped_partial

from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    format_3d_boundary,
    get_element_midpoints,
)
from modules.swarm_environments.mesh_refinement.problems.load_functions import (
    create_load_function,
)
from modules.swarm_environments.mesh_refinement.problems.load_functions.abstract_target_function import (
    AbstractTargetFunction,
)
from modules.swarm_environments.mesh_refinement.problems.problems_3d.abstract_finite_element_problem_3d import (
    AbstractFiniteElementProblem3D,
)

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_load(positions: np.array, load: AbstractTargetFunction):
    """
    Calculate the load x,y,z coordinates. This is the function "f" of the rhs of the poisson equation. It
    essentially defines where the "mass" of a system lies, and the solution of the poisson equation says in which
    direction the flow of gravity should be.
    A positive load means that we have positive mass, i.e., something that attracts.
    A negative load would correspond to a sink in something like fluid flow.
    For this load function, we consider loads that are Gaussian Mixture Models
    Args:
        positions: positions to evaluate as shape ( #points, 2)
        load: An AbstractTargetFunction instance that defines the GMM model that specifies the load

    Returns:

    """

    load_eval = load.evaluate(positions, include_gradient=False)
    return load_eval


def wrap_load(v, w, evaluate_load: callable, *args, **kwargs) -> np.ndarray:
    """
    Calculate the load for positions x and y. This is the function "f" of the rhs of the poisson equation.
    """
    x, y, z = w.x
    positions = np.stack((x, y, z), axis=-1)
    return evaluate_load(positions, *args, **kwargs) * v


class Poisson(AbstractFiniteElementProblem3D):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Initializes a Poisson equation with asymmetric load f(x,y)=x^c_x*y^c_y based on coefficients c_x and c_y.
        The result is a Poisson equation whose "center of mass" is parameterized by the given coefficients, with
        positive coefficients shifting the center in the positive direction and vice versa.
        Args:
            fem_config: Configuration for the finite element method. Contains
                domain: Dictionary for the (family of) problem domain(s)
                poisson: Dictionary consisting of keys for the specific kind of load function to build
            random_state: Internally used random_state to generate x and y coefficients. Will be used to new
                coefficients for every reset() call.
        """
        poisson_config = fem_config.get("poisson")
        self._load_function = create_load_function(
            target_function_name="gmm",
            target_function_config=poisson_config,
            fixed_target=poisson_config.get("fixed_load"),
            boundary=format_3d_boundary(fem_config.get("domain").get("boundary", None)),
            random_state=random_state,
            dimension=3,
        )
        super().__init__(fem_config=fem_config, random_state=random_state)  # also calls reset()
        from modules.swarm_environments.mesh_refinement.domains.domains_3d.cuboid import Cuboid

        assert isinstance(self._domain, Cuboid)

    def _set_pde(self) -> None:
        """
        Draw a new load function

        """
        self._load_function.reset(valid_point_function=self._points_in_domain)

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        mesh_ = mesh.with_boundaries(  # Label boundaries
            {
                "left": lambda x: np.isclose(x[0], self._boundary[0], atol=1e-10, rtol=1e-10),
                "right": lambda x: np.isclose(x[0], self._boundary[3], atol=1e-10, rtol=1e-10),
                "top": lambda x: np.isclose(x[1], self._boundary[1], atol=1e-10, rtol=1e-10),
                "bottom": lambda x: np.isclose(x[1], self._boundary[4], atol=1e-10, rtol=1e-10),
                "front": lambda x: np.isclose(x[2], self._boundary[2], atol=1e-10, rtol=1e-10),
                "back": lambda x: np.isclose(x[2], self._boundary[5], atol=1e-10, rtol=1e-10),
            },
            boundaries_only=True,
        )
        basis = Basis(mesh_, ElementTetP1())
        return basis

    def _points_in_domain(self, candidate_points: np.array) -> np.array:
        """
        Returns a subset of points that are inside the current domain, i.e., that can be found in the mesh.
        Returns:

        """
        corresponding_elements = self._domain.initial_mesh.element_finder()(*candidate_points.T)
        valid_points = candidate_points[corresponding_elements != -1]
        return valid_points

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
        K = asm(laplace, basis)
        f = asm(LinearForm(self.load), basis)  # rhs of the linear system that matches the load function

        mesh_interior = basis.complement_dofs(basis.get_dofs({"left", "right", "top", "bottom"}))
        # mesh_interior = basis.mesh.interior_nodes()
        K_interior, f_interior = condense(K, f, I=mesh_interior, expand=False)

        solution = self._solve_iterative(K_interior, f_interior, solution_shape=f.shape, mesh_interior=mesh_interior)
        return solution

    def _solve_iterative(self, K_interior, f_interior, solution_shape, mesh_interior) -> np.ndarray:
        """
        The below code implements an iterative solver with a diagonal preconditioning.
        A direct solver would simply look like this:
        solution = solve(K_interior, f_interior)
        Args:
            K_interior:
            solution_shape: Shape of the solution
            f_interior:
            mesh_interior:

        Returns:

        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import LinearOperator, cg, spsolve

        # Get the LinearOperator for the diagonal preconditioner
        diagonal_elements_interior = K_interior.diagonal()
        diagonal_elements_interior[diagonal_elements_interior == 0] = 1.0
        preconditioner_matrix_interior = diags(1.0 / diagonal_elements_interior, 0)
        M_interior = LinearOperator(
            matvec=lambda x: preconditioner_matrix_interior.dot(x),
            dtype=K_interior.dtype,
            shape=K_interior.shape,
        )
        # Call the Conjugate Gradient solver
        solution = np.zeros(solution_shape)
        solution[mesh_interior], info = cg(K_interior, f_interior, M=M_interior, tol=1e-10, maxiter=1000)
        # Fallback to a direct solver if CG did not converge
        if info > 0:
            solution = spsolve(K_interior, f_interior)
        return solution

    # wrapper functions for the load function for the finite element assembly
    @property
    def load(self) -> callable:
        return wrapped_partial(wrap_load, evaluate_load=evaluate_load, load=self._load_function)

    @property
    def load_function(self) -> callable:
        return wrapped_partial(evaluate_load, load=self._load_function)

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
        if "load_function" in element_feature_names:
            return np.array([self.load_function(get_element_midpoints(mesh))]).T
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
        return ["poisson"]

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_mesh(self, mesh: Mesh) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            mesh: The mesh to use for the feature calculation

        """
        additional_plots = {
            "load_function": self._get_scalar_plot(
                mesh=mesh, scalars=self.load_function(mesh.p.T), title="Load function"
            ),
            "log_load_function": self._get_scalar_plot(
                mesh=mesh,
                scalars=np.log(self.load_function(mesh.p.T) + 1e-8),
                title="Log load function",
            ),
        }
        return additional_plots
