r"""
Abstract Base class for Poisson equations.
The poisson equation is given as \Delta u = f, where \Delta is the Laplacian, u is the solution, and f is the
load. We consider a 2D domain with zero boundary conditions.
"""
import os
from typing import Any, Dict, Union

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    LinearForm,
    Mesh,
    asm,
    condense,
    solve,
)
from skfem.helpers import dot, grad

from modules.swarm_environments.mesh_refinement.problems.problems_2d.poisson import Poisson

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@BilinearForm
def laplace(u, v, _):
    # equivalent to `return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]`
    return dot(grad(u), grad(v))


def gaussian_inlet(x, mean, std):
    return np.exp(-((x - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi)) * np.sin(np.pi * x)


def map_and_project(boundary_positions, means, stds, scaler, rotation: float, full_circle: bool) -> np.array:
    """
    Maps each position on the boundary to a value between 0 and 1, and then evaluates a Gaussian function at that
    position.
    Args:
        boundary_positions:
        means: 2 Gaussian Means
        stds: 2 Gaussian Stds
        scaler:
        rotation:
        full_circle:

    Returns:

    """
    # x+np.sign(x)*y/(np.max(x)+np.max(y))
    progress = map_boundary_to_unit_value(boundary_positions)  # progress along the boundary for this section
    # if there is a 0 in progress, check if it should be 1 instead
    full_evaluations = np.zeros_like(progress)
    for mean, std in zip(means, stds):
        if full_circle:
            progress_ = (progress - mean) % 1  # "mean-shift" the progress to the interval [0, 1]
            evaluations = np.exp(-((progress_ - 0.5) ** 2) / (2 * std**2)) * scaler
        else:
            progress_ = (progress + rotation) % 1
            progress_ = (progress_ - np.min(progress_)) / (np.max(progress_) - np.min(progress_))  # normalize to [0, 1]
            evaluations = gaussian_inlet(progress_, mean, std) * scaler
        full_evaluations += evaluations
    return full_evaluations


def map_boundary_to_unit_value(points: np.array):
    x, y = points
    if np.any((x < 0) | (x > 1) | (y < 0) | (y > 1)):
        raise ValueError("All points must be on the border of the unit square")

    values = np.zeros_like(x)

    # Left side
    mask = x == 0
    values[mask] = 0.25 * y[mask]

    # Top side
    mask = y == 1
    values[mask] = 0.25 + 0.25 * x[mask]

    # Right side
    mask = x == 1
    values[mask] = 0.75 - 0.25 * y[mask]

    # Bottom side
    mask = np.logical_and(y == 0, x > 0)
    values[mask] = 1.0 - 0.25 * x[mask]

    # move progress in a modulo 1 ring

    return values


class GeneralizingPoisson(Poisson):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        poisson_config = fem_config.get("poisson")
        self._boundary_inlet_rate = poisson_config.get("boundary_inlet_rate", 0.95)
        self._corner_inlet_rate = poisson_config.get("corner_inlet_rate", 0.8)
        self._lower_random_scale = poisson_config.get("lower_random_scale", -1)

        self._pos_to_name = {
            0: "left",
            1: "bottom",
            2: "right",
            3: "top",
            4: "bottomleft",
            5: "bottomright",
            6: "topright",
            7: "topleft",
        }

        # all the randomness for the boundaries
        self._random_means = None
        self._random_stds = None
        self._random_scale = None
        self._inlet_at_boundary = None
        self._inlet_at_corner = None

        super().__init__(fem_config=fem_config, random_state=random_state)  # also calls reset()

    def _set_pde(self) -> None:
        """
        Draw a new load function

        """
        self._random_means = self._random_state.random(8)
        self._random_stds = self._random_state.random(8)  # np.exp(self._random_state.uniform(0, 1, 4))
        self._random_scale = np.exp(self._random_state.uniform(-10, self._lower_random_scale, 4))
        self._inlet_at_boundary = self._random_state.random(4) < self._boundary_inlet_rate
        self._inlet_at_corner = self._random_state.random(4) < self._corner_inlet_rate
        super()._set_pde()

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        By default, uses a linear triangular basis and no boundary on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        boundaries = {
            "left": lambda x: x[0] == self._boundary[0],
            "bottom": lambda x: x[1] == self._boundary[1],
            "right": lambda x: x[0] == self._boundary[2],
            "top": lambda x: x[1] == self._boundary[3],
            "bottomleft": lambda x: np.logical_and(x[0] == self._boundary[0], x[1] == self._boundary[1]),
            "bottomright": lambda x: np.logical_and(x[0] == self._boundary[2], x[1] == self._boundary[1]),
            "topright": lambda x: np.logical_and(x[0] == self._boundary[2], x[1] == self._boundary[3]),
            "topleft": lambda x: np.logical_and(x[0] == self._boundary[0], x[1] == self._boundary[3]),
        }
        mesh_ = mesh.with_boundaries(boundaries)
        return Basis(mesh_, ElementTriP1())

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        K = asm(laplace, basis)  # finite element assembly. Returns a sparse matrix
        f = asm(LinearForm(self.load), basis)  # rhs of the linear system that matches the load function

        inlets = [i for i, inlet in enumerate(self._inlet_at_boundary) if inlet]
        inlet_corners = [i for i, inlet in enumerate(self._inlet_at_corner) if inlet]
        groups = []

        # "join together" boundaries if they are adjacent.
        if len(inlets) == 4 and len(inlet_corners) >= 3:  # all boundaries are inlets, ignore the last corner
            groups.append([0, 1, 2, 3])
            boundaries = []
        else:
            # set boundaries
            boundaries = [i for i, inlet in enumerate(self._inlet_at_boundary) if not inlet]
            boundaries.extend([i + 4 for i, inlet in enumerate(self._inlet_at_corner) if not inlet])

            # check if there are groups of 3
            if 0 in inlets and 0 in inlet_corners and 1 in inlets and 1 in inlet_corners and 2 in inlets:
                groups.append([0, 1, 2])
            elif 1 in inlets and 1 in inlet_corners and 2 in inlets and 2 in inlet_corners and 3 in inlets:
                groups.append([1, 2, 3])
            elif 2 in inlets and 2 in inlet_corners and 3 in inlets and 3 in inlet_corners and 0 in inlets:
                groups.append([2, 3, 0])
            elif 3 in inlets and 3 in inlet_corners and 0 in inlets and 0 in inlet_corners and 1 in inlets:
                groups.append([3, 0, 1])
            else:
                if 0 in inlets and 0 in inlet_corners and 1 in inlets:
                    groups.append([0, 1])
                if 1 in inlets and 1 in inlet_corners and 2 in inlets:
                    groups.append([1, 2])
                if 2 in inlets and 2 in inlet_corners and 3 in inlets:
                    groups.append([2, 3])
                if 3 in inlets and 3 in inlet_corners and 0 in inlets:
                    groups.append([3, 0])

            # get singletons
            for inlet in inlets:
                if not any(inlet in group for group in groups):
                    groups.append([inlet])
        # Define the inlet basis
        mesh = basis.mesh
        K_lil = K.tolil()

        for position, group in enumerate(groups):
            inlet_dofs = basis.get_dofs([self._pos_to_name[i] for i in group])
            boundary_positions = mesh.p[:, inlet_dofs.nodal_ix]
            projected_solutions = map_and_project(
                boundary_positions,
                self._random_means[2 * position : 2 * position + 2],
                stds=self._random_stds[2 * position : 2 * position + 2],
                scaler=self._random_scale[position],
                rotation=0.25 * group[0],
                full_circle=len(group) == 4,
            )
            # scaler=self._random_scale[position])

            # Set the corresponding rows to zero in the LIL matrix
            K_lil[inlet_dofs.nodal_ix, :] = 0

            # Set the diagonal entries to 1 in the LIL matrix
            K_lil[inlet_dofs.nodal_ix, inlet_dofs.nodal_ix] = 1

            f[inlet_dofs.nodal_ix] = projected_solutions

        # Convert K back to CSR format
        K = K_lil.tocsr()

        if len(boundaries) > 0:
            boundaries = [self._pos_to_name[i] for i in boundaries]
            condensed_system = condense(K, f, D=basis.get_dofs(boundaries))
            solution = solve(*condensed_system)
        else:
            solution = solve(K, f)
        return solution
