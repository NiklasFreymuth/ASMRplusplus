from typing import Any, Dict, Type, Union

import numpy as np

from modules.swarm_environments.mesh_refinement.problems.problems_2d.abstract_finite_element_problem_2d import (
    AbstractFiniteElementProblem,
)


def create_finite_element_problem(
    *, fem_config: Dict[Union[str, int], Any], random_state: np.random.RandomState
) -> AbstractFiniteElementProblem:
    """
    Builds and returns a finite element problem class.
    Args:
        fem_config: Config containing additional details about the finite element method. Contains
            poisson: Config containing additional details about the poisson problem. Depends on the problem
            domain: Config containing additional details about the domain. Depends on the domain
        random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
            function class

    Returns: Some domain class that inherits from AbstractDomain.

    """
    pde = get_finite_element_problem_class(fem_config=fem_config)
    return pde(fem_config=fem_config, random_state=random_state)


def get_finite_element_problem_class(*, fem_config: Dict[Union[str, int], Any]) -> Type[AbstractFiniteElementProblem]:
    """
    Builds and returns a finite element problem class.
    Args:
        fem_config: Config containing additional details about the finite element method.


    Returns: Some domain class that inherits from AbstractDomain.

    """

    pde_type = fem_config.get("pde_type")
    pde_type = pde_type.lower()
    if pde_type in ["laplace"]:
        from modules.swarm_environments.mesh_refinement.problems.problems_2d.laplace import (
            Laplace,
        )

        pde = Laplace
    elif pde_type == "poisson":
        dimension = fem_config.get("domain", {}).get("dimension", 2)
        if dimension == 2:
            boundary_type = fem_config.get("poisson", {}).get("boundary_type", "zero")
            if boundary_type == "zero":
                from modules.swarm_environments.mesh_refinement.problems.problems_2d.poisson import (
                    Poisson,
                )

                pde = Poisson
            elif boundary_type == "general":
                from modules.swarm_environments.mesh_refinement.problems.problems_2d.generalizing_poisson import (
                    GeneralizingPoisson,
                )

                pde = GeneralizingPoisson
            elif boundary_type == "neumann":
                from modules.swarm_environments.mesh_refinement.problems.problems_2d.neumann_poisson import (
                    NeumannPoisson,
                )

                pde = NeumannPoisson
        elif dimension == 3:
            from modules.swarm_environments.mesh_refinement.problems.problems_3d.poisson import (
                Poisson,
            )

            pde = Poisson
        else:
            raise ValueError(f"Poisson problem for {dimension} dimensions is not supported.")
    elif pde_type == "stokes_flow":
        from modules.swarm_environments.mesh_refinement.problems.problems_2d.stokes_flow import (
            StokesFlow,
        )

        pde = StokesFlow
    elif pde_type == "linear_elasticity":
        from modules.swarm_environments.mesh_refinement.problems.problems_2d.linear_elasticity import (
            LinearElasticity,
        )

        pde = LinearElasticity
    elif pde_type == "heat_diffusion":
        last_step_only = fem_config.get("heat_diffusion", {}).get("last_step_only", False)
        if last_step_only:
            from modules.swarm_environments.mesh_refinement.problems.problems_2d.last_step_heat_diffusion import (
                LastStepHeatDiffusion,
            )

            pde = LastStepHeatDiffusion
        else:
            from modules.swarm_environments.mesh_refinement.problems.problems_2d.heat_diffusion import (
                HeatDiffusion,
            )

            pde = HeatDiffusion

    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")
    return pde
