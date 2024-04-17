from typing import Any, Dict, List, Optional, Union

import numpy as np

from modules.swarm_environments.mesh_refinement.evaluation import (
    EvaluationFEMProblemCircularQueue,
)
from modules.swarm_environments.mesh_refinement.fem_problem_wrapper import FEMProblemWrapper


class EvaluationFEMProblemStream(EvaluationFEMProblemCircularQueue):
    def __init__(
        self,
        *,
        fem_config: Dict[Union[str, int], Any],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        super().__init__(fem_config=fem_config, random_state=random_state)
        num_pdes = fem_config.get("num_pdes")  # number of pdes to store. None, 0 or -1 means infinite
        self._fem_problems = None
        self._fem_problem_seeds: List[Optional[int]] = [None for _ in range(num_pdes)]

    def _next_from_idx(self, pde_idx: int):
        if self._fem_problem_seeds[pde_idx] is None:
            # draw a new seed for this fem Problem
            self._fem_problem_seeds[pde_idx] = self._random_state.randint(0, 2**31)
        new_seed = self._fem_problem_seeds[pde_idx]
        new_problem = self._fem_problem_class(
            fem_config=self._fem_config,
            random_state=np.random.RandomState(seed=new_seed),
        )
        new_problem = FEMProblemWrapper(
            fem_config=self._fem_config,
            fem_problem=new_problem,
            pde_features=self._pde_features,
        )
        new_problem.reset()
        return new_problem

    @property
    def num_pdes(self):
        return self._max_index
