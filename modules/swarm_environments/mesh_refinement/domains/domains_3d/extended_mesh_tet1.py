from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Type, Union

import numpy as np
from skfem import Element, MeshTet1


@dataclass(repr=False)
class ExtendedMeshTet1(MeshTet1):
    """
    A wrapper/extension of the Scikit FEM MeshTri1 that allows for more flexible mesh initialization.
    This class allows for arbitrary sizes and centers of the initial meshes, and offers utility for different initial
    mesh types.

    """

    @classmethod
    def init_truncated_cube(
        cls: Type,
        max_element_volume: float = 0.01,
        initial_meshing_method: str = "custom",
        intersection_point=np.array([0.5, 0.5, 0.5]),
        *args,
        **kwargs,
    ) -> MeshTet1:
        """
        Creates a truncated cube by intersecting three planes at the point intersection_point. and removing the union
        of the half-planes on positive side of each plane

        Args:
            max_element_volume: Maximum volume of a single element in the mesh.
            initial_meshing_method: Either "meshpy" or "custom". "meshpy" uses the meshpy library to generate an initial
                mesh. "custom" uses a custom algorithm to generate an initial mesh.
            intersection_point: Position where the three planes intersect to create the truncated cube.
            *args:
            **kwargs:

        Returns:

        """

        # Create lists for x, y, and z points
        p_x = [0, intersection_point[0], 1]
        p_y = [0, intersection_point[1], 1]
        p_z = [0, intersection_point[2], 1]

        mesh = MeshTet1.init_tensor(p_x, p_y, p_z)

        # "cut off" part of the mesh based on intersection_point
        mask_nodes = np.all(mesh.p > intersection_point[:, None], axis=0)
        nodes_to_remove = np.where(mask_nodes)[0]
        mask_elements = np.isin(mesh.t, nodes_to_remove)
        elements_to_keep = np.all(~mask_elements, axis=0)
        new_elements = mesh.t[:, elements_to_keep]

        # Mask out old nodes and reindex elements
        mask_nodes_keep = ~np.isin(np.arange(mesh.p.shape[1]), nodes_to_remove)
        new_nodes = np.where(mask_nodes_keep)[0]
        reindex_map = np.full(mesh.p.shape[1], -1, dtype=int)
        reindex_map[new_nodes] = np.arange(len(new_nodes))

        new_p = mesh.p[:, mask_nodes_keep]
        reindexed_elements = reindex_map[new_elements]

        mesh = cls(new_p, reindexed_elements)

        if initial_meshing_method == "meshpy":
            import meshpy.tet as tetgen

            # Use tetgen to create the mesh
            mesh_info = tetgen.MeshInfo()
            mesh_info.set_points(mesh.p.T)
            mesh_info.set_facets(mesh.facets.T)
            mesh = tetgen.build(mesh_info, max_volume=max_element_volume)

            # Extract the nodes and elements from the meshpy result
            new_p = np.array(mesh.points).T
            new_elements = np.array(mesh.elements).T
            mesh = cls(new_p, new_elements)
        elif initial_meshing_method == "custom":
            pass  # custom mesh is already created
        else:
            raise ValueError(f"Unknown initial meshing method {initial_meshing_method}")

        return mesh

    @classmethod
    def init_cuboid(
        cls: Type,
        max_element_volume: float = 0.01,
        initial_meshing_method: str = "custom",
        lengths: np.array = np.array([1, 1, 1]),
        *args,
        **kwargs,
    ) -> MeshTet1:
        """
        Creates a unit cube mesh

        Args:
            max_element_volume: Maximum volume of a single element in the mesh.
            initial_meshing_method: Either "meshpy" or "custom". "meshpy" uses the meshpy library to generate an initial
                mesh. "custom" uses a custom algorithm to generate an initial mesh.
            lengths: Lengths of the cuboid in x, y, and z direction
            *args:
            **kwargs:

        Returns:

        """
        # Create lists for x, y, and z points
        lengths = lengths / np.max(lengths)  # normalize lengths to 1
        if initial_meshing_method == "meshpy":
            import meshpy.tet as tetgen

            p_x = [0, lengths[0]]
            p_y = [0, lengths[1]]
            p_z = [0, lengths[2]]
            mesh = MeshTet1.init_tensor(p_x, p_y, p_z)

            # Use tetgen to create the mesh
            mesh_info = tetgen.MeshInfo()
            mesh_info.set_points(mesh.p.T)
            mesh_info.set_facets(mesh.facets.T)
            mesh = tetgen.build(mesh_info, max_volume=max_element_volume)

            # Extract the nodes and elements from the meshpy result
            new_p = np.array(mesh.points).T
            new_elements = np.array(mesh.elements).T
            mesh = cls(new_p, new_elements)
        elif initial_meshing_method == "custom":
            if max_element_volume is not None:
                # Adjust segment length based on desired max volume of tetrahedral elements
                segment_length = np.minimum((6 * max_element_volume) ** (1 / 3), np.min(lengths))
                num_segments = np.maximum((lengths / segment_length).astype(int), 1)
            else:
                # Default behavior with a fixed increment
                num_segments = ((lengths / np.min(lengths)) + 1).astype(int)

            p_x = np.linspace(0, lengths[0], num_segments[0] + 1)
            p_y = np.linspace(0, lengths[1], num_segments[1] + 1)
            p_z = np.linspace(0, lengths[2], num_segments[2] + 1)
            mesh = ExtendedMeshTet1.init_tensor(p_x, p_y, p_z)

        else:
            raise ValueError(f"Unknown initial meshing method {initial_meshing_method}")

        return mesh

    def refined(self, times_or_ix: Union[int, np.ndarray] = 1):
        """Return a refined mesh.

        Parameters
        ----------
        times_or_ix
            Either an integer giving the number of uniform refinements or an
            array of element indices for adaptive refinement.

        """
        m = self
        if isinstance(times_or_ix, int):
            for _ in range(times_or_ix):
                m = m._uniform()
        else:
            m = m._adaptive(times_or_ix)
        return m

    def element_finder(self, mapping=None):
        """
        Find the element that contains the points [(x, y, z)]. Returns -1 if the point is in no element
        Args:
            mapping: A mapping from the global node indices to the local node indices. Currently not used

        Returns:

        """
        from pykdtree.kdtree import KDTree

        from modules.swarm_environments.util.point_in_3d_geometry import points_in_tetrahedra

        nelems = self.t.shape[1]
        elements = self.p[:, self.t].T
        tree = KDTree(np.mean(self.p[:, self.t], axis=1).T)

        fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

        def finder(x, y, z, _k=1, _brute_force=False):
            """
            For each point in (x,y,z), find the element that contains this point.
            Returns -1 for points that are not in any element
            Args:
                x: Array of point x coordinates of shape (num_points, )
                y: Array of point y coordinates of shape (num_points, )
                z: Array of point z coordinates of shape (num_points, )
                _k: Number of elements to consider for each point. Only used if _brute_force is False.
                   Will do a first pass over k=5, another one for k=10, then k=20 and finally fall back to brute force
                _brute_force: If True, use the brute force variant. If False, use the KDTree to select candidates.
                Internal parameter that is used to switch between the two algorithms as a backup if the KDTree fails.

            Returns:

            """
            if _brute_force:
                # brute force approach - check all elements
                element_indices = points_in_tetrahedra(
                    points=np.array([x, y, z]).T,
                    tetrahedra=elements,
                    candidate_indices=None,
                )
            else:
                # find candidate elements
                num_candidates = min(fibonacci[_k], nelems)

                # use the KDTree to find the elements with the closest center
                distances, candidate_indices = tree.query(np.array([x, y, z]).T, num_candidates)
                # usually (distance, index), but we only care about the indices

                if _k > 1:
                    # only use the last half as the previous half was considered in the previous iteration
                    candidate_indices = candidate_indices[:, fibonacci[_k - 1] :]
                candidate_indices = candidate_indices.astype(np.int64)  # cast to int64 for compatibility reasons

                # try to find the right element for each point using the KDTree candidates
                element_indices = points_in_tetrahedra(
                    points=np.array([x, y, z]).T,
                    tetrahedra=elements,
                    candidate_indices=candidate_indices,
                )

                # fallback to brute force search for elements that were not found in the KDTree
                invalid_elements = element_indices == -1
                if invalid_elements.any():
                    if _k <= 6:
                        element_indices[invalid_elements] = finder(
                            x=x[invalid_elements],
                            y=y[invalid_elements],
                            z=z[invalid_elements],
                            _k=_k + 1,
                            _brute_force=False,
                        )
                    else:
                        element_indices[invalid_elements] = finder(
                            x=x[invalid_elements],
                            y=y[invalid_elements],
                            z=z[invalid_elements],
                            _brute_force=True,
                        )

            return element_indices

        return finder

    def __post_init__(self):
        """
        Copied over to remove warning
        """
        if self.sort_t:
            self.t = np.sort(self.t, axis=0)

        self.doflocs = np.asarray(self.doflocs, dtype=np.float64, order="K")
        self.t = np.asarray(self.t, dtype=np.int64, order="K")

        M = self.elem.refdom.nnodes

        if self.nnodes > M and self.elem is not Element:
            # reorder DOFs to the expected format: vertex DOFs are first
            # note: not run if elem is not set
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = np.arange(len(uniq), dtype=np.int64)[ix].reshape(t_nodes.shape)
            doflocs = np.hstack(
                (
                    p[:, uniq],
                    np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
                )
            )
            doflocs[:, self.dofs.element_dofs[M:].flatten("F")] = p[:, t[M:].flatten("F")]
            self.doflocs = doflocs

        # C_CONTIGUOUS is more performant in dimension-based slices
        if not self.doflocs.flags["C_CONTIGUOUS"]:
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags["C_CONTIGUOUS"]:
            self.t = np.ascontiguousarray(self.t)

    def save(
        self,
        filename: Union[str, PathLike],
        point_data: Optional[Dict[str, np.ndarray]] = None,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The output filename, with suffix determining format;
            e.g. .msh, .vtk, .xdmf
        point_data
            Data related to the vertices of the mesh.
        cell_data
            Data related to the elements of the mesh.

        """
        from skfem.io.meshio import to_file

        return to_file(MeshTet1(self.p, self.t), filename, point_data, cell_data, **kwargs)


###############################################################################
#                                                                             #
#                              Test Cases                                     #
#                                                                             #
###############################################################################


def _test_truncated_cube_poisson():
    # simple test case for a poisson problem. Tests the visualization capabilities and the element finder
    import plotly.graph_objects as go
    from skfem import Basis, ElementTetP1, asm, condense, solve, solver_iter_pcg
    from skfem.models.poisson import laplace, unit_load

    from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
        get_tetrahedron_volumes_from_indices,
    )
    from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_3d import (
        get_3d_interpolation_traces,
        get_3d_scatter_mesh_traces,
    )

    mesh = ExtendedMeshTet1.init_truncated_cube(
        max_element_volume=1,
        initial_meshing_method="meshpy",
        intersection_point=np.array([0.2, 0.4, 0.5]),
        # intersection_point=np.array([0.5, 0.5, 0.5]),
    )
    mesh = mesh.refined(2)
    # for i in range(5):
    #     mesh = mesh.refined([0, 1, 2, 3])
    # solve a laplace problem on the mesh for a nicer visualization
    basis = Basis(mesh, ElementTetP1())
    A = asm(laplace, basis)
    b = asm(unit_load, basis)
    mesh_interior = basis.mesh.interior_nodes()
    A_interior, b_interior = condense(A, b, I=mesh_interior, expand=False)
    x = np.zeros_like(b)
    x[mesh_interior] = solve(
        A_interior,
        b_interior,
        solver=solver_iter_pcg(verbose=False, M=None),
    )
    import plotly.io as pio

    pio.renderers.default = "browser"
    print(mesh)
    traces = get_3d_scatter_mesh_traces(mesh, scalars=x, showlegend=True, size=5, opacity=1)
    fig = go.Figure(data=traces)
    fig.show()
    traces = get_3d_interpolation_traces(mesh, scalars=x, showlegend=True)
    fig = go.Figure(data=traces)
    fig.show()
    volumes = get_tetrahedron_volumes_from_indices(mesh.p.T, mesh.t.T)
    traces = get_3d_scatter_mesh_traces(mesh, scalars=volumes, showlegend=True, colorbar_title="Volume")
    fig = go.Figure(data=traces)
    fig.show()


def main():
    # simple test case for a poisson problem. Tests the visualization capabilities and the element finder
    import plotly.graph_objects as go

    from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_3d import (
        get_3d_interpolation_traces,
    )

    # _test_truncated_cube_poisson()
    mesh = ExtendedMeshTet1.init_cuboid(
        max_element_volume=1,
        initial_meshing_method="meshpy",
        # intersection_point=np.array([0.5, 0.5, 0.5]),
    )
    for i in range(5):
        mesh = mesh.refined([0])
    from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization_3d import (
        get_3d_wireframe_trace,
    )

    traces = [get_3d_wireframe_trace(mesh, showlegend=True)]
    fig = go.Figure(data=traces)
    fig.show()
    print(mesh)
    #
    # mesh2 = mesh.refined([0])
    # mesh3 = mesh.refined([1])
    # traces = get_3d_wireframe_trace(mesh2, showlegend=True)
    # fig = go.Figure(data=traces)
    # fig.show()
    # print(mesh2)
    #
    # traces = get_3d_wireframe_trace(mesh3, showlegend=True)
    # fig = go.Figure(data=traces)
    # fig.show()
    # print(mesh3)

    traces = get_3d_interpolation_traces(mesh, scalars=np.arange(mesh.t.shape[1]), showlegend=True)
    fig = go.Figure(data=traces)
    fig.show()


if __name__ == "__main__":
    main()
