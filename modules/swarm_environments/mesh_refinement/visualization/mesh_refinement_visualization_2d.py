from typing import List

import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Mesh

"""
This class handles the 2d visualization for the MeshRefinement environment.
While there is built-in visualization utility in scikit-FEM, it uses matplotlib as a backend and is thus
unfortunately not compatible with our framework and wandb.
We instead re-write the important parts of it here using plotly.
"""


def contour_trace_from_element_values(
    mesh: Mesh, scalars: np.array, trace_name: str = "Value per Agent"
) -> List[BaseTraceType]:
    """
    Creates a list of plotly traces for the elements of a mesh
    Args:
        mesh: A scikit-fem mesh
        scalars: A numpy array containing scalar evaluations of the mesh per mesh element or mesh node
        trace_name: Name/Title of the trace
    Returns:
        A list of plotly traces

    """

    if scalars.shape[0] == mesh.nelements:
        intensitymode = "cell"
    elif scalars.shape[0] == mesh.nvertices:
        intensitymode = "vertex"
    else:
        raise ValueError(
            f"Invalid shape for scalars. Must be ({mesh.nvertices},) or ({mesh.nelements},), given {scalars.shape}"
        )
    vertex_positions = mesh.p.T
    faces = mesh.t.T

    if vertex_positions.shape[-1] == 2:
        z = np.zeros_like(vertex_positions[:, 0])
    else:
        z = vertex_positions[:, 2]
    face_trace = go.Mesh3d(
        x=vertex_positions[:, 0],
        y=vertex_positions[:, 1],
        z=z,
        flatshading=True,  # Enable flat shading
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        text=scalars,
        intensity=scalars,
        intensitymode=intensitymode,
        cmin=np.nanmin(scalars),
        cmax=np.nanmax(scalars),
        colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"),  # put colorbar on the left
        colorscale="Jet",
        name=trace_name,
        hovertemplate="v: %{intensity:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<extra></extra>",
        # + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>",
        showlegend=True,
    )
    traces = [face_trace]
    return traces


def get_mesh_traces(mesh: Mesh, color: str = "black", showlegend: bool = True) -> List[BaseTraceType]:
    """
    Draws a plotly trace depicting the edges/facets of a scikit fem triangle mesh
    Args:
        mesh: A scikit basis. Contains a basis.mesh attribute that has properties
         * mesh.facets of shape (2, num_edges) that lists indices of edges between the mesh, and
         * mesh.p of shape (2, num_nodes) for coordinates between those indices
        color: Color of scatter plot
        showlegend: Whether to show the legend

    Returns: A list of plotly traces [mesh_trace, node_trace], where mesh_trace consists of the outlines of the mesh
        and node_trace consists of an overlay of all nodes

    """
    facets = mesh.facets
    vertices = mesh.p

    node_trace = go.Scatter3d(
        x=vertices[0],
        y=vertices[1],
        z=np.zeros_like(vertices[0]),
        mode="markers",
        marker={"size": 0.7, "color": color},
        name="Vertices",
        showlegend=showlegend,
    )

    num_edges = facets.shape[-1]
    edge_x_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_z_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_x_positions[0::3] = vertices[0, facets[0]]
    edge_x_positions[1::3] = vertices[0, facets[1]]
    edge_y_positions[0::3] = vertices[1, facets[0]]
    edge_y_positions[1::3] = vertices[1, facets[1]]

    edge_z_positions[0::3] = np.zeros_like(vertices[0, facets[0]])
    edge_z_positions[1::3] = np.zeros_like(vertices[0, facets[1]])
    edge_trace = go.Scatter3d(
        x=edge_x_positions,
        y=edge_y_positions,
        z=edge_z_positions,
        mode="lines",
        line=dict(color=color, width=1),
        name="Wireframe",
        showlegend=showlegend,
    )
    return [edge_trace, node_trace]
