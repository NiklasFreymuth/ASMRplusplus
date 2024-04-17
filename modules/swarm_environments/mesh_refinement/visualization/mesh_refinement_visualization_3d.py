from typing import List

import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Mesh

"""
This class handles the 3d visualization for the MeshRefinement environment.
While there is built-in visualization utility in scikit-FEM, it uses matplotlib as a backend and is thus
unfortunately not compatible with our framework and wandb.
We instead re-write the important parts of it here using plotly.
"""


def get_3d_interpolation_traces(
    mesh: Mesh,
    scalars: np.array,
    intensitymode: str = "vertex",
    colorbar_title: str = "Scalars",
    opacity: float = 0.3,
    wireframe_opacity: float = 0.2,
    showlegend: bool = True,
) -> List[BaseTraceType]:
    """
    Generates Plotly traces for visualizing a scalar field interpolated over a 3D scikit-fem mesh.

    This function returns a list of Plotly traces representing the mesh as a colored 3D surface
    interpolated by the provided scalar field. It includes additional traces for mesh vertices
    and the wireframe.

    Args:
        mesh (Mesh): A scikit-fem Mesh object. Should contain the attributes `mesh.p` with shape
            (3, num_vertices) for vertex coordinates and `mesh.facets` for face definitions.
        scalars (array-like): An array of shape (num_vertices,) or (num_faces,)
        containing the scalar field to be plotted.
        intensitymode (str, optional): Determines if the scalar field is interpolated over the
            vertices or the faces of the mesh. Defaults to "vertex". For faces, use "cell".
        colorbar_title (str, optional): Title for the colorbar. Defaults to "Potential".
        opacity (float, optional): The opacity level of the mesh surface. Defaults to 0.3.
        wireframe_opacity (float, optional): The opacity level of the mesh wireframe. Defaults to 0.2.
        showlegend (bool, optional): Determines if the legend should be shown in the plot. Defaults to True.

    Returns:
        List[BaseTraceType]: A list of Plotly traces, including:
            - One trace for the interpolated mesh surface.
            - Additional traces for the vertices and the wireframe.
    """
    assert intensitymode in [
        "vertex",
        "cell",
    ], f"Invalid intensitymode {intensitymode}. Must be 'vertex' or 'cell'."
    assert scalars.shape[0] in [
        mesh.nvertices,
        mesh.nelements,
    ], f"Invalid shape for scalars. Must be ({mesh.nvertices},) or ({mesh.nelements},), given {scalars.shape}"
    mesh_trace = go.Mesh3d(
        x=mesh.p[0],
        y=mesh.p[1],
        z=mesh.p[2],
        i=mesh.facets[0],
        j=mesh.facets[1],
        k=mesh.facets[2],
        intensity=scalars,
        intensitymode=intensitymode,
        colorscale="Viridis",
        opacity=opacity,  # needs to be small to see through all surfaces
        colorbar=dict(
            title=colorbar_title,
            yanchor="top",
            y=1,
            x=0,
            ticks="outside",  # put colorbar on the left
        ),
        showlegend=showlegend,
        name="Value (v)",
        hovertemplate="v: %{intensity:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>",
    )
    traces = [
        mesh_trace,
        _get_vertices_3d_trace(mesh, showlegend=showlegend),
        get_3d_wireframe_trace(mesh, opacity=wireframe_opacity, showlegend=showlegend),
    ]
    return traces


def get_3d_scatter_mesh_traces(
    mesh: Mesh,
    scalars,
    colorbar_title: str = "Scalars",
    opacity: float = 0.8,
    wireframe_opacity: float = 0.2,
    size: int = 3,
    showlegend: bool = True,
) -> List[BaseTraceType]:
    """
    Generates Plotly traces for a 3D scatter plot over a scikit-fem mesh.

    This function returns a list of Plotly traces representing the mesh vertices
    as point markers colored by a given scalar field and a wireframe overlay.

    Args:
        mesh (Mesh): A scikit-fem Mesh object. Should contain the attribute `mesh.p`
            with shape (3, num_vertices), holding coordinates of vertices.
        scalars (array-like): An array of shape (num_vertices,) or (num_elements containing
            the scalar field to be plotted. If num_elements, the element midpoints are used.
        colorbar_title (str, optional): Title for the colorbar. Defaults to "Potential".
        opacity (float, optional): The opacity level of the scatter plot. Defaults to 0.8.
        wireframe_opacity (float, optional): The opacity level of the mesh wireframe. Defaults to 0.2.
        size (int, optional): The size of the scatter markers. Defaults to 1.
        showlegend (bool, optional): Determines if legend should be shown in the plot. Defaults to True.

    Returns:
        List[BaseTraceType]: A list of Plotly traces, including:
            - One trace representing the colored scatter plot.
            - Additional traces for the mesh wireframe.
    """
    if scalars.shape[0] == mesh.nelements:
        positions = mesh.p[:, mesh.t].mean(axis=1)
    elif scalars.shape[0] == mesh.nvertices:
        positions = mesh.p
    else:
        raise ValueError(
            f"Invalid shape for scalars. " f"Must be ({mesh.nelements},)  or ({mesh.nvertices},), given {scalars.shape}"
        )

    scatter_trace = get_3d_scatter_trace(positions, scalars, colorbar_title, opacity, showlegend, size)
    wireframe_trace = get_3d_wireframe_trace(mesh, opacity=wireframe_opacity, showlegend=showlegend)
    traces = [scatter_trace, wireframe_trace]
    # no node trace because this is contained in the scatter plot
    return traces


def get_3d_scatter_trace(
    positions,
    scalars,
    colorbar_title,
    opacity: float = 0.8,
    showlegend: bool = True,
    size: int = 1,
):
    scatter_trace = go.Scatter3d(
        x=positions[0],
        y=positions[1],
        z=positions[2],
        mode="markers",
        showlegend=showlegend,
        marker=dict(
            size=size,
            color=scalars,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=opacity,
            colorbar=dict(
                title=colorbar_title,
                yanchor="top",
                y=1,
                x=0,
                ticks="outside",  # put colorbar on the left
            ),
        ),
        hovertemplate="v: %{marker.color:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>",
        name="Value (v)",
    )
    return scatter_trace


def get_3d_wireframe_trace(mesh: Mesh, showlegend: bool, opacity: float = 0.2, color: str = "black") -> BaseTraceType:
    """
    Generates Plotly traces for the edges/the wireframe of a 3D scikit-fem mesh.

    This function returns a list of Plotly traces that contain the wireframe of a 3D mesh.
    The wireframe consists of unique edges connecting the mesh vertices.

    Args:
        mesh (Mesh): A scikit-fem Mesh object. Should contain attributes:
            - `mesh.p`: Coordinates of vertices, shape (3, num_vertices).
            - `mesh.t`: Indices of vertices per element, shape (4, num_elements) for tetrahedra.
        opacity (float, optional): The opacity level of the wireframe. Defaults to 0.5.
        showlegend (bool, optional): Determines if legend should be shown in the plot. Defaults to True.
        color (str, optional): The color of the mesh and nodes in the Plotly traces. Defaults to 'black'.

    Returns:
        List[BaseTraceType]: A list containing a single Plotly trace representing the mesh edges/the mesh wireframe.
    """
    vertices = mesh.p
    edges = mesh.edges
    num_edges = edges.shape[1]
    edge_x_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_z_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_x_positions[0::3] = vertices[0, edges[0]]
    edge_x_positions[1::3] = vertices[0, edges[1]]
    edge_y_positions[0::3] = vertices[1, edges[0]]
    edge_y_positions[1::3] = vertices[1, edges[1]]
    edge_z_positions[0::3] = vertices[2, edges[0]]
    edge_z_positions[1::3] = vertices[2, edges[1]]

    wireframe_trace = go.Scatter3d(
        x=edge_x_positions,
        y=edge_y_positions,
        z=edge_z_positions,
        mode="lines",
        line=dict(color=color, width=1),
        name="Wireframe",
        showlegend=showlegend,
        opacity=opacity,
    )

    return wireframe_trace


def _get_vertices_3d_trace(mesh: Mesh, showlegend: bool, opacity=0.8, color: str = "black") -> BaseTraceType:
    """
    Generates a Plotly trace for the vertices of a 3D scikit-fem mesh.

    This function returns a list containing a single Plotly trace representing
    the mesh vertices as point markers.

    Args:
        mesh (Mesh): A scikit-fem Mesh object. Should contain the attribute `mesh.p`
            with shape (3, num_vertices), holding coordinates of vertices.
        showlegend (bool): Determines if legend should be shown in the plot.
        opacity (float, optional): The opacity level of the vertex markers. Defaults to 0.8.
        color (str, optional): The color of the vertex markers. Defaults to 'black'.

    Returns:
        BaseTraceType: A Plotly trace that represents the vertices.
    """
    node_trace = go.Scatter3d(
        x=mesh.p[0],
        y=mesh.p[1],
        z=mesh.p[2],
        mode="markers",
        name="Vertices",
        showlegend=showlegend,
        marker=dict(
            size=1,
            opacity=opacity,
            color=color,
        ),
    )
    return node_trace
