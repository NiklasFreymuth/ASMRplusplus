r"""Fluid Flow task.
Solves the Poiseuille flow problem (?) using the Navier-Stokes equations.
Here, we are interested in the steady state solution, i.e., the solution that is independent of time.
As such, the domain must always be a trapezoid, and the inlet velocity is parabolic.

We compute both the velocity and pressure fields, and weight their contributions to the reward function.
Further, we use quadratic elements for the velocity field, and linear elements for the pressure field.
"""

import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from plotly import graph_objects as go
from skfem import (
    Basis,
    DiscreteField,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    Mesh,
    asm,
    bmat,
    condense,
    solve,
)
from skfem.models.general import divergence, rot
from skfem.models.poisson import laplace, vector_laplace

from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    get_element_midpoints,
    get_line_segment_distances,
)
from modules.swarm_environments.mesh_refinement.problems.problems_2d.abstract_finite_element_problem_2d import (
    AbstractFiniteElementProblem2D,
)

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parabolic_inlet_profile(
    inlet_coordinates: DiscreteField,
    parabolic_velocity: float,
    minimum_y_value: float,
    maximum_y_value: float,
    phase: float = np.pi,
) -> np.array:
    """
    Define the parabolic velocity profile at the inlet. Higher values of the velocity will result in a more
    turbulent flow, and thus a longer time to reach steady state.
    Args:
        inlet_coordinates:  The inlet coordinates. A discrete field of shape ([x,y], inlet_facets, quadrature_points)
        parabolic_velocity: The maximum velocity at the inlet
        minimum_y_value:    The minimum y value of the inlet
        maximum_y_value:    The maximum y value of the inlet
        phase: The phase of the sinusoidal component of the inlet velocity profile. Should be 0 or pi, because the
          velocity profile on the boundaries should be 0.

    Returns: The plane Poiseuille parabolic inlet profile, i.e.,
    a discrete field of shape ([u,v], inlet_facets, quadrature_points) where u and v are the x and y components of the
    velocity respectively, and v = 0.
    The equation for this profile reads u = velocity * y * (1 - y) + sin(2 * pi * y)
    The first term is the parabolic profile, the second term is the sinusoidal component. The phase of the sinusoidal
    component is given by the phase argument.

    """
    x_inlet_coordinates, y_inlet_coordinates = _normalize_inlet_coordinates(
        inlet_coordinates, maximum_y_value, minimum_y_value
    )

    velocity_profile = parabolic_velocity * y_inlet_coordinates * (1.0 - y_inlet_coordinates) + np.sin(
        phase + 2 * np.pi * y_inlet_coordinates
    )
    return np.stack([velocity_profile, np.zeros_like(x_inlet_coordinates)])


def gaussian_inlet_profile(
    inlet_coordinates: DiscreteField,
    mean: float,
    std: float,
    minimum_y_value: float,
    maximum_y_value: float,
    phase: float = np.pi,
) -> np.array:
    x_inlet_coordinates, y_inlet_coordinates = _normalize_inlet_coordinates(
        inlet_coordinates, maximum_y_value, minimum_y_value
    )

    def gaussian(x, mean, std):
        return np.exp(-((x - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

    velocity_profile = gaussian(y_inlet_coordinates, mean=mean, std=std) * np.sin(
        phase + 2 * np.pi * y_inlet_coordinates
    )
    return np.stack([velocity_profile, np.zeros_like(x_inlet_coordinates)])


def _normalize_inlet_coordinates(inlet_coordinates, maximum_y_value, minimum_y_value):
    # normalize inlet coordinates to adapt to inlets that are not in x=[0,1]
    x_inlet_coordinates = inlet_coordinates[0]
    y_inlet_coordinates = inlet_coordinates[1]
    y_inlet_coordinates = (y_inlet_coordinates - minimum_y_value) / (maximum_y_value - minimum_y_value)
    return x_inlet_coordinates, y_inlet_coordinates


def solve_velocity_and_pressure(
    velocity_basis: Basis,
    minimum_y_value: float,
    maximum_y_value: float,
    inlet_type: str,
    inlet_features: Dict[str, float],
    phase: float = np.pi,
) -> Tuple[np.array, np.array]:
    """
    Solve the Navier-Stokes equations for the velocity and pressure fields

    Args:
        velocity_basis: The basis for the velocity field
        minimum_y_value:    The minimum y value of the inlet
        maximum_y_value:    The maximum y value of the inlet
        inlet_type: The type of inlet profile. Can be either "parabolic" or "gaussian"
        inlet_features: A dictionary containing the features of the inlet. The following keys are possible:
            parabolic_velocity: The maximum velocity at the inlet if the inlet profile is parabolic
            gaussian_mean: The mean of the gaussian if the inlet profile is gaussian
            gaussian_std: The standard deviation of the gaussian if the inlet profile is gaussian
        phase: The phase of the sinusoidal component of the inlet velocity profile. Should be 0 or pi, because the
            velocity profile on the boundaries should be 0.

    Returns: The velocity and pressure fields as np.arrays of shape
    (velocity_basis.dofs,) and (pressure_basis.dofs,) respectively

    """
    mesh = velocity_basis.mesh
    element = velocity_basis.elem

    # Create the pressure basis, which is a scalar basis on the same mesh as the velocity basis, but with elements
    # of a lower order
    pressure_basis = velocity_basis.with_element(ElementTriP1())

    # Assemble the block matrix K
    A = asm(vector_laplace, velocity_basis)
    B = -asm(divergence, velocity_basis, pressure_basis)
    K = bmat([[A, B.T], [B, None]], "csr")
    # # block matrix with the following structure
    # # | A  B^T |
    # # | B   0  |
    # # CSR stands for Compressed Sparse Row format

    # Define the inlet basis
    inlet_basis = FacetBasis(mesh, element, facets=mesh.boundaries["inlet"])

    # Solve the system of equations
    if inlet_type == "parabolic":
        parabolic_velocity = inlet_features["parabolic_velocity"]
        parabolic_ = partial(
            parabolic_inlet_profile,
            parabolic_velocity=parabolic_velocity,
            phase=phase,
            minimum_y_value=minimum_y_value,
            maximum_y_value=maximum_y_value,
        )
    elif inlet_type == "gaussian":
        gaussian_mean = inlet_features["gaussian_mean"]
        gaussian_std = inlet_features["gaussian_std"]
        parabolic_ = partial(
            gaussian_inlet_profile,
            mean=gaussian_mean,
            std=gaussian_std,
            phase=phase,
            minimum_y_value=minimum_y_value,
            maximum_y_value=maximum_y_value,
        )
    else:
        raise ValueError(f"Inlet type must be either 'parabolic' or 'gaussian', given '{inlet_type}'")
    velocity_inlet_basis = inlet_basis.project(parabolic_)

    uvp = np.hstack(
        (
            velocity_inlet_basis,
            pressure_basis.zeros(),
        )
    )
    # remove the right side of the boundary, i.e., the outlet. The boundary here goes over the facets
    boundary_dofs = velocity_basis.dofs.get_facet_dofs(
        velocity_basis.mesh.facets_satisfying(lambda x: x[0] < 1, boundaries_only=True)
    )

    uvp = solve(*condense(K, x=uvp, D=boundary_dofs))
    # Separate the velocity and pressure components
    velocity, pressure = np.split(uvp, [A.shape[0]])
    return velocity, pressure


def get_stream_function(velocity_basis: Basis, velocity: np.array) -> np.array:
    """
    Calculate the stream field from the velocity field
    Args:
        velocity_basis: The basis for the velocity field
        velocity: The velocity field

    Returns: The stream function on the nodal points of the mesh

    """
    # Define the basis for the stream function
    stream_basis = velocity_basis.with_element(ElementTriP1())

    A = asm(laplace, stream_basis)
    vorticity = asm(rot, stream_basis, w=velocity_basis.interpolate(velocity))
    psi = solve(*condense(A, vorticity, D=stream_basis.get_dofs("tube")))
    # psi is the stream function, which is the solution to the Laplace equation of the vorticity field
    return psi[stream_basis.nodal_dofs].squeeze()


class StokesFlow(AbstractFiniteElementProblem2D):
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
        # displacement of the right boundary
        stokes_flow_config = fem_config.get("stokes_flow")

        self._fixed_inlet = stokes_flow_config.get("fixed_inlet")
        self._inlet_type = stokes_flow_config.get("inlet_type")

        # define a range of available velocities from the range specified in the config
        self._parabolic_velocity_range = None
        self._gaussian_mean_range = None
        self._gaussian_std_range = None
        if self._inlet_type == "parabolic":
            self._parabolic_velocity_range = np.array(
                [
                    stokes_flow_config.get("lower_velocity"),
                    stokes_flow_config.get("upper_velocity"),
                ]
            )
        elif self._inlet_type == "gaussian":
            self._gaussian_mean_range = np.array(
                [
                    0.5 - stokes_flow_config.get("mean_range"),
                    0.5 + stokes_flow_config.get("mean_range"),
                ]
            )
            self._gaussian_std_range = np.array(
                [
                    stokes_flow_config.get("lower_std"),
                    stokes_flow_config.get("upper_std"),
                ]
            )
        else:
            raise ValueError(f"Unknown inlet type '{self._inlet_type}'")

        self._inlet_features = None

        # elements and meshes
        self.velocity_element_vector = ElementVector(ElementTriP2(), dim=2)
        # the element for the velocity needs to be an order of magnitude higher than the element for the pressure.
        # This is known as a Taylor-Hood element, and is necessary for the stability of the Navier-Stokes equations
        # See also: https://www.math.colostate.edu/~bangerth/videos/676/slides.33.25.pdf)

        # cache
        self._vertex_velocities = None

        self._lower_left_y = None
        self._upper_left_y = None

        super().__init__(fem_config=fem_config, random_state=random_state)  # also calls reset() and thus _set_pde()

    def _set_pde(self) -> None:
        """
        Draw a new PDE instance from the available family of plate bending PDEs.

        """
        # set the inlet stream
        self._set_inlet()

        # assert the domain and pre-compute some values for the boundary conditions
        self._set_boundaries()

    def _set_inlet(self):
        if self._inlet_type == "parabolic":
            if self._fixed_inlet:
                parabolic_velocity = np.mean(self._parabolic_velocity_range)
            else:
                # log-uniform sampling of the velocity, since changes in the velocity are more significant at low velocities
                log_velocity_range = np.log(self._parabolic_velocity_range)
                parabolic_velocity = np.exp(
                    self._random_state.uniform(low=log_velocity_range[0], high=log_velocity_range[1], size=1).item()
                )
            self._inlet_features = {"parabolic_velocity": parabolic_velocity}
        elif self._inlet_type == "gaussian":
            if self._fixed_inlet:
                gaussian_mean = np.mean(self._gaussian_mean_range)
                gaussian_std = np.mean(self._gaussian_std_range)
            else:
                # log-uniform sampling of the std, since changes in the std are more significant at low std
                gaussian_mean = self._random_state.uniform(
                    low=self._gaussian_mean_range[0],
                    high=self._gaussian_mean_range[1],
                    size=1,
                ).item()
                log_std_range = np.log(self._gaussian_std_range)
                gaussian_std = np.exp(
                    self._random_state.uniform(low=log_std_range[0], high=log_std_range[1], size=1).item()
                )
            self._inlet_features = {
                "gaussian_mean": gaussian_mean,
                "gaussian_std": gaussian_std,
            }
        else:
            raise ValueError(f"Unknown inlet type '{self._inlet_type}'")

    def _set_boundaries(self):
        boundary_nodes = self._domain.boundary_nodes  # normalized corners of the trapezoid domain
        lower_left_y = boundary_nodes[0, 1]
        upper_left_y = boundary_nodes[1, 1]
        self._lower_left_y = lower_left_y
        self._upper_left_y = upper_left_y
        self._boundaries = {
            "inlet": lambda x: np.isclose(x[0], 0, atol=1e-10, rtol=1e-10),
        }

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        By default, uses a linear triangular basis and no boundary on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        mesh_ = mesh.with_boundaries(self._boundaries)

        # Define basis with integration order 2 for the velocity field
        basis = Basis(mesh_, self.velocity_element_vector, intorder=2)
        return basis

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution for the fluid flow problem.
        Args:
            basis: The basis to use for the solution.
            cache: Whether to cache the solution for plotting purposes.

        Returns: An array (num_dofs, 3), where every entry corresponds to a vector of the norm of the velocity
        at the corresponding dof, and x, y, components of the velocity.

        """

        velocity, pressure = solve_velocity_and_pressure(
            velocity_basis=basis,
            inlet_type=self._inlet_type,
            inlet_features=self._inlet_features,
            minimum_y_value=self._lower_left_y,
            maximum_y_value=self._upper_left_y,
        )

        vertex_velocity = velocity[basis.nodal_dofs]  # velocity on the vertices, i.e., a solution per vertex
        if cache:
            self._vertex_velocities = vertex_velocity
        else:
            self._vertex_velocities = None

        velocity = velocity.reshape(2, -1, order="F")

        velocity_norm = np.linalg.norm(velocity, axis=0)
        # stream = get_stream_function(basis, velocity)
        solution = np.hstack(
            (
                velocity_norm[:, None],
                velocity[0, ..., None],  # x component of the velocity
                velocity[1, ..., None],  # y component of the velocity
            )
        )
        return solution

    @property
    def has_different_nodal_solution(self) -> bool:
        return True

    @property
    def nodal_solution(self) -> np.array:
        """
        Retrieves the nodal solution, i.e., the solution on just the mesh nodes.
        Args:

        Returns: An array (num_vertices, 3), where every entry corresponds to a vector of the norm of the velocity
        at the corresponding vertex, and x, y, components of the velocity.

        """
        vertex_velocity = self._vertex_velocities  # velocity on the vertices, i.e., a solution per vertex
        velocity_norm = np.linalg.norm(vertex_velocity, axis=0)
        # stream = get_stream_function(basis, velocity)
        solution = np.hstack(
            (
                velocity_norm[:, None],
                vertex_velocity[0, ..., None],
                vertex_velocity[1, ..., None],
            )
        )
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

        features = []
        if "velocity" in element_feature_names:
            num_elements = mesh.nelements
            if "parabolic_velocity" in self._inlet_features:
                parabolic_velocity = np.repeat(
                    np.array(self._inlet_features["parabolic_velocity"]),
                    repeats=num_elements,
                    axis=0,
                )
            else:
                parabolic_velocity = np.zeros((num_elements, 1))
            features.append(parabolic_velocity)

        if "distance_to_inlet" in element_feature_names:
            distance_to_inlet = get_line_segment_distances(
                points=get_element_midpoints(mesh),
                projection_segments=self.inlet_line_segments,
                return_minimum=True,
            )
            features.append(distance_to_inlet)

        return np.array(features).T if len(features) > 0 else None


    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["velocity_norm", "velocity_x", "velocity_y"]

    def project_to_scalar(self, values: np.array) -> np.array:
        """
        Projects a value per node and solution dimension to a scalar value per node.
        Equivalent to np.einsum('ij, j -> i', self._solution, self.fem_problem.solution_dimension_weights
        if simple weights are used
        Args:
            values: A vector of shape ([num_vertices,] solution_dimension)

        Returns: A scalar value per vertex
        """
        return np.sqrt(values[..., 1] ** 2 + values[..., 2] ** 2) / np.sqrt(2)

    @property
    def solution_dimension_weights(self) -> np.array:
        """
        Returns a list of weights for the solution dimensions. This is used to weight the solution dimensions
        when calculating the error.
        Returns: A list of weights for the solution dimensions that sums up to one

        """
        raise NotImplementedError("Solution dimension weights not implemented for FluidFlow equation")

    @property
    def inlet_line_segments(self):
        source_facets = self.initial_mesh.facets_satisfying(lambda x: x[0] == 0, boundaries_only=True)  # inlet facets
        boundary_node_indices = self.initial_mesh.facets[:, source_facets]
        line_segments = self.initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        return line_segments

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_mesh(self, mesh: Mesh) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            mesh: The mesh to use for the feature calculation

        """
        if self._vertex_velocities is not None:
            mesh_x_velocity = np.mean(self._vertex_velocities[0][mesh.t], axis=0)
            mesh_y_velocity = np.mean(self._vertex_velocities[1][mesh.t], axis=0)

            additional_plots = {
                "velocity_x": self._get_scalar_plot(mesh=mesh, scalars=mesh_x_velocity, title="Velocity x"),
                "velocity_y": self._get_scalar_plot(mesh=mesh, scalars=mesh_y_velocity, title="Velocity y"),
            }
            return additional_plots
        else:
            return {}
