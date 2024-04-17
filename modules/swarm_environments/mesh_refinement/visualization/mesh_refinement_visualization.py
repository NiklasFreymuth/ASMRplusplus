import numpy as np
from skfem import Mesh

from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_2d import (
    contour_trace_from_element_values,
    get_mesh_traces,
)
from modules.swarm_environments.util.visualization import get_layout

def get_plotly_mesh_traces_and_layout(
    mesh: Mesh,
    scalars: np.ndarray,
    mesh_dimension: int,
    title: str,
    boundary: np.ndarray = None,
    scatter: bool = True,
):
    """

    Args:
        mesh: A scikit-fem mesh
        scalars: The scalar values to plot per element or per node of the mesh
        mesh_dimension: The dimension of the mesh
        title:
        boundary:
        scatter: If 3d, whether to show the solutions as a scatter or as a surface

    Returns:

    """
    if mesh_dimension == 2:
        element_midpoint_trace = contour_trace_from_element_values(
            mesh=mesh,
            scalars=scalars.flatten(),
            trace_name="Scalar",
        )
        mesh_trace = get_mesh_traces(mesh)
        traces = element_midpoint_trace + mesh_trace
        if boundary is None:
            boundary = np.concatenate((mesh.p.min(axis=1), mesh.p.max(axis=1)), axis=0)
        layout = get_layout(boundary=boundary, title=title, layout_type="mesh2d")

    elif mesh_dimension == 3:
        if scatter:
            from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_3d import (
                get_3d_scatter_mesh_traces,
            )

            traces = get_3d_scatter_mesh_traces(mesh=mesh, scalars=scalars)
        else:
            from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_3d import (
                get_3d_interpolation_traces,
            )

            traces = get_3d_interpolation_traces(mesh=mesh, scalars=scalars)
        layout = get_layout(title=title, layout_type="mesh3d")
    else:
        raise ValueError(f"Unknown mesh dimension {mesh.dim()}")
    return traces, layout
