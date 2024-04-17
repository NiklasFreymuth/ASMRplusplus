import math

import numpy as np
from skfem import Mesh
from skfem.visuals.matplotlib import draw, plot


def get_element_midpoints(mesh: Mesh, transpose: bool = True) -> np.array:
    """
    Get the midpoint of each element
    Args:
        mesh: The mesh as a skfem.Mesh object
        transpose: Whether to transpose the result. If True
            the result will be of shape (num_elements, 2), if False, it will be of shape (2, num_elements). Defaults
            to True.

    Returns: Array of shape (num_elements, 2)/(2, num_elements) containing the midpoint of each element

    """
    midpoints = np.mean(mesh.p[:, mesh.t], axis=1)
    if transpose:
        return midpoints.T
    else:
        return midpoints
    # return np.mean(mesh.p[:, mesh.t], axis=1).T


def visualize_solution(mesh: Mesh, solution: np.array, draw_mesh: bool = True):
    if draw_mesh:
        ax = draw(mesh)
        return plot(mesh, solution, ax=ax, shading="gouraud", colorbar=True)
    else:
        return plot(mesh, solution, shading="gouraud", colorbar=True)


def convert_error(error_estimate, element_indices, num_vertices: int):
    """
    get error per vertex from error per element

    Args:
        error_estimate: Error estimate per element
        element_indices: Elements of the mesh. Array of shape (num_elements, 3)
        num_vertices: Number of vertices of the mesh
    Returns:

    """
    error_per_node = np.zeros(shape=(num_vertices,))
    np.add.at(error_per_node, element_indices, error_estimate[:, None])  # inplace operation
    return error_per_node


def get_aggregation_per_element(
    solution: np.array,
    element_indices: np.array,
    aggregation_function_str: str = "mean",
) -> np.array:
    """
    get aggregation of solution per element from solution per vertex by adding all spanning vertices for each element

    Args:
        solution: Error estimate per element of shape (num_elements, solution_dimension)
        element_indices: Elements of the mesh. Array of shape (num_elements, vertices_per_element),
        where vertices_per_element is 3 triangular meshes
        aggregation_function_str: The aggregation function to use. Can be 'mean', 'std', 'min', 'max', 'median'
    Returns: An array of shape (num_elements, ) containing the solution per element

    """
    if aggregation_function_str == "mean":
        solution_per_element = solution[element_indices].mean(axis=1)
    elif aggregation_function_str == "std":
        solution_per_element = solution[element_indices].std(axis=1)
    elif aggregation_function_str == "min":
        solution_per_element = solution[element_indices].min(axis=1)
    elif aggregation_function_str == "max":
        solution_per_element = solution[element_indices].max(axis=1)
    elif aggregation_function_str == "median":
        solution_per_element = np.median(solution[element_indices], axis=1)
    else:
        raise ValueError(f"Aggregation function {aggregation_function_str} not supported")
    return solution_per_element


def angle_to(point: np.ndarray, center_point: np.ndarray = np.array([0.5, 0.5])):
    """
    get angle to a center point for a given point
    center point is mapped to the origin

    Args:
        point: An array of shape [2, ]
        center_point: An array of shape [2, ]
    Returns:
        returns a scalar value - the angle to the center point
    """

    x = point[0] - center_point[0]
    y = point[1] - center_point[1]

    return math.atan2(y, x)


def get_line_segment_distances(
    points: np.array,
    projection_segments: np.array,
    return_minimum: bool = False,
    return_tangent_points: bool = False,
) -> np.array:
    """
    Calculates the distances of an array of points to an array of line segments.
    Vectorized for any number of points and line segments
    Args:
        points: An array of shape [num_points, 2], i.e., an array of points to project towards the projection segments
        projection_segments: An array of shape [num_segments, 4], i.e., an array of line segments/point pairs
        return_minimum: If True, the minimum distance is returned. If False, an array of all distances is returned
        return_tangent_points: If True, distances and tangent points of the projections to all segments are returned

    Returns: An array of shape [num_points, {num_segments, 1}] containing the distance of each point to each segment
        or the minimum segment, depending on return_minimum

    """
    segment_distances = projection_segments[:, :2] - projection_segments[:, 2:]
    tangent_positions = np.sum(projection_segments[:, :2] * segment_distances, axis=1) - points @ segment_distances.T
    segment_lengths = np.linalg.norm(segment_distances, axis=1)

    # the normalized tangent position is in [0,1] if the projection to the line segment is directly possible
    normalized_tangent_positions = tangent_positions / segment_lengths**2

    # it gets clipped to [0,1] otherwise, i.e., clips projections to the boundary of the line segment.
    # this is necessary since line segments may describe an internal part of the mesh domain, meaning
    # that we always want the distance to the segment rather than the distance to the line it belongs to
    normalized_tangent_positions[normalized_tangent_positions > 1] = 1  # clip too big values
    normalized_tangent_positions[normalized_tangent_positions < 0] = 0  # clip too small values
    tangent_points = projection_segments[:, :2] - normalized_tangent_positions[..., None] * segment_distances
    projection_vectors = points[:, None, :] - tangent_points

    distances = np.linalg.norm(projection_vectors, axis=2)
    if return_minimum:
        distances = np.min(distances, axis=1)
    if return_tangent_points:
        return distances, tangent_points
    return distances


def get_simplex_volumes_from_indices(positions: np.array, simplex_indices: np.array) -> np.array:
    """
    Computes the volume for an array of simplices.
    Args:
    positions: Array of shape (#points, 3) of (x,y,z) coordinates
    simplex_indices: Array of shape (#simplices, 3) containing point indices that span simplices
    Returns: An array of shape (#simplices,) of volumes for the input simplices
    """
    if positions.shape[-1] == 2:  # 2d case:
        return get_triangle_areas_from_indices(positions=positions, triangle_indices=simplex_indices)
    elif positions.shape[-1] == 3:  # 3d case:
        return get_tetrahedron_volumes_from_indices(positions=positions, tetrahedron_indices=simplex_indices)
    else:
        raise ValueError(f"Cannot compute simplex volumes for {positions.shape[-1]} dimensions")


def get_tetrahedron_volumes_from_indices(positions: np.array, tetrahedron_indices: np.array) -> np.array:
    """
    Computes the volume for an array of tetrahedra.
    Args:
    positions: Array of shape (#points, 3) of (x,y,z) coordinates
    tetrahedron_indices: Array of shape (#tetrahedra, 4) containing point indices that span tetrahedra
    Returns: An array of shape (#tetrahedra,) of volumes for the input tetrahedra
    """
    # Extract coordinates of tetrahedron vertices
    v0 = positions[tetrahedron_indices[:, 0]]
    v1 = positions[tetrahedron_indices[:, 1]]
    v2 = positions[tetrahedron_indices[:, 2]]
    v3 = positions[tetrahedron_indices[:, 3]]

    # Compute the volume
    volume = np.abs(np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0)
    return volume


def get_triangle_areas_from_indices(positions: np.array, triangle_indices: np.array) -> np.array:
    """
    Computes the area for an array of triangles using the triangle-wise formula
    Area = 0.5*| (Xb-Xa)(Yc-Ya)-(Xc-Xa)(Yb-Ya) | where a,b,c are 3 vertices
    for coordinates X and Y
    Args:
        positions: Array of shape (#points, 2) of (x,y) coordinates
        triangle_indices: Array of shape (#triangles, 3) containing point indices that span triangles

    Returns: An array of shape (#triangles,) of areas for the input triangles

    """

    area = np.abs(
        0.5
        * (
            (positions[triangle_indices[:, 1], 0] - positions[triangle_indices[:, 0], 0])
            * (positions[triangle_indices[:, 2], 1] - positions[triangle_indices[:, 0], 1])
            - (positions[triangle_indices[:, 2], 0] - positions[triangle_indices[:, 0], 0])
            * (positions[triangle_indices[:, 1], 1] - positions[triangle_indices[:, 0], 1])
        )
    )
    return area


def format_3d_boundary(boundary):
    if boundary is None:
        boundary = np.array([0, 0, 0, 1, 1, 1])
    elif len(boundary) == 3:
        boundary = np.array([0, 0, 0, *boundary])
    elif len(boundary) == 6:
        boundary = np.array(boundary)
    else:
        raise ValueError(f"Invalid boundary {boundary}. Must be of length 3 or 6.")
    return boundary


def sample_in_range(max_value: float, min_value: float, sampling_type: str) -> float:
    """
    Generates a sample value within a specified range using different sampling distributions.
    Args:
        max_value: The maximum value of the range.
        min_value: The minimum value of the range.
        sampling_type: The type of sampling distribution to use.
            Can be 'uniform', 'loguniform', 'beta', or 'logbeta'.

    Returns: A sample value generated based on the specified distribution.

    Raises:
        ValueError: If an unknown sampling_type is provided.

    Note:
        'uniform' generates a sample from a uniform distribution.
        'loguniform' generates a sample from a log-uniform distribution.
        'beta' generates a sample from a beta distribution scaled to the range.
        'logbeta' generates a sample from a beta distribution in log-space, scaled to the range.
    """
    if sampling_type == "uniform":
        # Sample from a uniform distribution over [min_value, max_value].
        return np.random.uniform(min_value, max_value)

    elif sampling_type == "loguniform":
        # Sample from a log-uniform distribution over [min_value, max_value].
        log_min = np.log(min_value)
        log_max = np.log(max_value)
        return np.exp(np.random.uniform(log_min, log_max))

    elif sampling_type == "beta":
        # Sample from a beta distribution (alpha=beta=0.5) scaled to [min_value, max_value].
        sample = np.random.beta(0.5, 0.5)
        return sample * (max_value - min_value) + min_value

    elif sampling_type == "logbeta":
        # Sample from a beta distribution (alpha=beta=0.5) in log-space, scaled to [min_value, max_value].
        sample = np.random.beta(0.5, 0.5)
        log_min = np.log(min_value)
        log_max = np.log(max_value)
        return np.exp(sample * (log_max - log_min) + log_min)

    else:
        raise ValueError(f"Unknown sampling type {sampling_type}")
