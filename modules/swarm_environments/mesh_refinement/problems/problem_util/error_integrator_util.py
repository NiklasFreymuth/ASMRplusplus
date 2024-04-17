from typing import Optional

import numpy as np
import torch
import torch_scatter
from numpy import ndarray
from skfem import Basis

from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    get_simplex_volumes_from_indices,
)


def get_integration_weights(basis: Basis, scale_integration_by_area: bool) -> np.array:
    """
    Calculates the integration weights for the given mesh.
    Args:
        basis: Basis. Contains the (triangular) mesh to calculate the integration weights for.
        scale_integration_by_area: Whether to scale the integration weights by the area of the corresponding

    Returns: The integration weights for the given mesh.

    """
    if scale_integration_by_area:
        integration_weights = get_simplex_volumes_from_indices(positions=basis.mesh.p.T, simplex_indices=basis.mesh.t.T)
    else:
        integration_weights = np.ones(basis.mesh.t.shape[1])
    integration_weights = integration_weights / np.sum(integration_weights)  # normalize

    return integration_weights


def find_integration_points_on_boundary(
    coarse_basis,
    integration_points: np.array,
    corresponding_elements: Optional[np.array] = None,
    dim: int = 2,
) -> np.array:
    """
    Finds for each integration point whether it lies on a boundary between two elements and if so, which elements
    correspond to the edge.
    Args:
        coarse_basis: The basis of the coarse mesh to calculate the error estimate for based on the fine integration
            mesh.
        integration_points: The integration points of the fine integration mesh.
            An array of shape (num_integration_points, [2,3])
        corresponding_elements: (Optional) The elements of the coarse mesh that contain the integration points. Used to
            determine the elements that share the boundary that contains the integration points without having to do
            a full search.
            An array of shape (num_integration_points,) containing the indices of the elements.
        dim: The dimension of the problem. Can be 2 or 3.
    Returns: An array of boundary indices with shape (num_points, ) containing the index of the boundary
        that contains the point. If the point does not lie on any boundary, the index is -1.

    """
    edges = coarse_basis.mesh.facets
    boundary_positions = coarse_basis.mesh.p[:, edges].T

    if corresponding_elements is not None:
        candidate_indices = coarse_basis.mesh.t2f.T[corresponding_elements]
    else:
        candidate_indices = None

    if dim == 2:
        from modules.swarm_environments.util.point_in_2d_geometry import points_on_edges

        boundary_membership = points_on_edges(
            points=integration_points.T,
            edges=boundary_positions,
            candidate_indices=candidate_indices,
        )
    elif dim == 3:
        from modules.swarm_environments.util.point_in_3d_geometry import points_on_faces

        boundary_membership = points_on_faces(
            points=integration_points.T,
            faces=boundary_positions,
            candidate_indices=candidate_indices,
        )
    else:
        raise ValueError(f"Unknown dimension: {dim}")
    # array of edge indices with shape (num_points, ) containing the index of the edge that contains the point.
    return boundary_membership


def get_integrated_differences(
    pointwise_differences: np.array,
    integration_weights: np.array,
    corresponding_boundary_elements: np.array,
    corresponding_full_elements: np.array,
    valid_boundary_indices: np.array,
    num_coarse_elements: int,
    error_metric: str,
) -> np.array:
    """
    Calculates the integrated differences between the coarse evaluation and the reference evaluation.
    For this, the differences are summed up over the corresponding coarse elements by summing over
    integration points on boundaries between two elements and integration points inside elements.
    Works for 2d and 3d meshes.
    Args:
        pointwise_differences: The absolute error/difference per integration point.
            Shape (#integration points, #solution dimensions)
        integration_weights: The weights of the integration points. Shape (#integration points,)
        corresponding_boundary_elements: The elements of the coarse mesh which have edges with integration points on
        them. Array of shape (#integration_points_on_edges, {2,3}), as each edge is between two elements in 2d and
        between three elements in 3d
        corresponding_full_elements: The elements of the coarse mesh which have integration points on them.
            Array of shape (#integration_points_on_elements,)
        valid_boundary_indices: The indices of the integration points that lie on boundaries between two elements.
            Array of shape (#integration_points_on_edges,).
            Can be used to get the indices of the integration points that lie inside elements.
        num_coarse_elements: The number of faces/elements of the coarse mesh.
        error_metric: The metric to use for calculating the differences between the coarse evaluation and
            the reference evaluation. Can be "mean" or "squared".
        dim: The dimension of the problem. Can be 2 or 3.


    Returns: elementwise_differences, which are the integrated differences between the coarse evaluation and the
        reference evaluation per *coarse* element as an array of shape (#elements, #solution dimensions).

    """
    assert error_metric in [
        "mean",
        "squared",
        "maximum",
    ], f"Unknown error metric: {error_metric}"

    if "squared" in error_metric:
        pointwise_differences = pointwise_differences**2

    if "maximum" in error_metric:
        scatter_operation = lambda *args, **kwargs: torch_scatter.scatter_max(*args, **kwargs)[0]
    else:
        scatter_operation = torch_scatter.scatter_sum
        pointwise_differences = pointwise_differences * integration_weights[:, None]

    # sum-aggregate over corresponding faces by adding up the differences of the integration points
    # differences when mapping the current_basis to the reference_basis' integration points for each point
    # we do this in torch because it is a lot faster than numpy
    pointwise_differences = torch.tensor(pointwise_differences)
    corresponding_boundary_elements = torch.tensor(corresponding_boundary_elements)
    corresponding_full_elements = torch.tensor(corresponding_full_elements)

    boundary_differences = pointwise_differences[valid_boundary_indices]
    in_element_differences = pointwise_differences[~valid_boundary_indices]
    left_boundary_differences = scatter_operation(
        boundary_differences,
        corresponding_boundary_elements[:, 0],
        dim=0,
        dim_size=num_coarse_elements,
    )
    right_boundary_differences = scatter_operation(
        boundary_differences,
        corresponding_boundary_elements[:, 1],
        dim=0,
        dim_size=num_coarse_elements,
    )
    element_differences = scatter_operation(
        in_element_differences,
        corresponding_full_elements,
        dim=0,
        dim_size=num_coarse_elements,
    )

    if "maximum" in error_metric:
        elementwise_difference = torch.max(left_boundary_differences, right_boundary_differences)
        elementwise_difference = torch.max(elementwise_difference, element_differences)
    else:
        elementwise_difference = ((left_boundary_differences + right_boundary_differences) / 2) + element_differences
    elementwise_difference = elementwise_difference.numpy()

    return elementwise_difference


def probes_from_elements(basis: Basis, x: ndarray, cells: ndarray):
    """
    Return matrix which acts on a solution vector to find its values. Uses pre-computed cell indices.
    on points `x`.
    Args:
        basis: The basis to use for the interpolation
        x: The points to interpolate to
        cells: Cell indices per point. A cell index of -1 means that the point is not in any cell.

    Returns:

    """
    import sys

    if "pyodide" in sys.modules:
        from scipy.sparse.coo import coo_matrix
    else:
        from scipy.sparse import coo_matrix
    pts = basis.mapping.invF(x[:, :, np.newaxis], tind=cells)
    phis = np.array([basis.elem.gbasis(basis.mapping, pts, k, tind=cells)[0] for k in range(basis.Nbfun)]).flatten()
    return coo_matrix(
        (
            phis,
            (
                np.tile(np.arange(x.shape[1]), basis.Nbfun),
                basis.element_dofs[:, cells].flatten(),
            ),
        ),
        shape=(x.shape[1], basis.N),
    )
