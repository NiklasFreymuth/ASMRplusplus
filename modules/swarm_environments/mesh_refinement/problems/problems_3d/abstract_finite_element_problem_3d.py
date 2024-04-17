r"""
Base class for an Abstract (Static) Finite Element Problem.
The problem specifies a partial differential equation to be solved, and the boundary conditions. It also specifies the
domain/geometry of the problem.
Currently, uses a triangular mesh with linear elements.
"""
import os
from abc import ABC
from typing import Any, Dict, Union

import numpy as np
import plotly.graph_objects as go
from skfem import Basis, ElementTetP1, Mesh

from modules.swarm_environments.mesh_refinement.mesh_refinement_util import format_3d_boundary
from modules.swarm_environments.mesh_refinement.problems.abstract_finite_element_problem import (
    AbstractFiniteElementProblem,
)

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AbstractFiniteElementProblem3D(AbstractFiniteElementProblem, ABC):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        boundary = fem_config.get("domain").get("boundary", None)
        self._boundary: np.ndarray = format_3d_boundary(boundary)
        super().__init__(fem_config=fem_config, random_state=random_state)

    def _assert_consistency(self) -> None:
        from modules.swarm_environments.mesh_refinement.domains.domains_3d import (
            AbstractDomain3D,
        )

        assert isinstance(
            self._domain, AbstractDomain3D
        ), f"Domain dimension is {type(self._domain)}, but should be AbstractDomain3D"

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        return Basis(mesh, ElementTetP1())

    ###############################
    # plotting utility functions #
    ###############################

    def _get_scalar_plot(self, mesh: Mesh, scalars: np.ndarray, title: str = "Value") -> go.Figure:
        """
        Plot the given scalar values on the current mesh.
        Args:
            mesh: The 3d scikit FEM mesh to use for the plot and the wireframe
            scalars: A value for each vertex or element of the mesh
            title: The title of the plot

        Returns: A plotly figure

        """
        from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization import (
            get_plotly_mesh_traces_and_layout,
        )

        traces, layout = get_plotly_mesh_traces_and_layout(
            mesh=mesh, scalars=scalars, mesh_dimension=3, title=title, boundary=None
        )
        return go.Figure(data=traces, layout=layout)
