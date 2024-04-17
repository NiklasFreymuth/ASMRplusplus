from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.swarm_environments import MeshRefinement
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.swarm_environments.util.keys import ELEMENT_PENALTY_LAMBDA, NUM_AGENTS
from modules.swarm_environments.util.torch_util import detach


def _single_rollout(
    environment: MeshRefinement,
    policy_step_function: Callable,
    normalized_position: float,
) -> List[Dict[str, Any]]:
    """

    Args:
        environment:
        policy_step_function:
        normalized_position: The "Position" of the rollout between 0 and 1.
            Used to set e.g., a fixed element penalty lambda.

    Returns:

    """

    # reset environment and prepare loop over rollout
    _ = environment.reset()
    environment.set_element_penalty_lambda(normalized_position)
    observation = environment.last_observation  # reset observation since the element penalty was changed
    done = False
    full_additional_information = []
    while not done:
        actions, values = policy_step_function(observation=observation)
        actions = detach(actions)
        observation, reward, done, additional_information = environment.step(action=actions)
        full_additional_information.append(additional_information)
    return full_additional_information


def get_results_from_additional_information(last_additional_information):
    num_agents = last_additional_information.get(NUM_AGENTS)
    top_error = last_additional_information.get("top_error")
    top01_error = last_additional_information.get("top0.1_error")
    mean_error = last_additional_information.get("mean_error")
    squared_error = last_additional_information.get("squared_error")
    top1_error = last_additional_information.get("top1_error")
    top5_error = last_additional_information.get("top5_error")
    top10_error = last_additional_information.get("top10_error")
    log_num_agents = np.log(num_agents + 1e-16)
    log_mean_error = np.log(mean_error + 1e-16)
    log_squared_error = np.log(squared_error + 1e-16)
    log_top_error = np.log(top_error + 1e-16)
    log_top01_error = np.log(top01_error + 1e-16)
    element_penalty_lambda = last_additional_information.get(ELEMENT_PENALTY_LAMBDA)
    current_results = [
        num_agents,
        top_error,
        top01_error,
        mean_error,
        squared_error,
        top1_error,
        top5_error,
        top10_error,
        log_num_agents,
        log_mean_error,
        log_squared_error,
        log_top_error,
        log_top01_error,
        element_penalty_lambda,
    ]
    return current_results


def _create_dataframe(experiment_results: np.array):
    num_penalties, num_pdes, _ = experiment_results.shape

    # Reshape the results to store all individual results in the dataframe
    data = experiment_results.reshape(num_penalties * num_pdes, -1)

    # Create a dataframe with multi-index for num_penalties and num_pdes
    index = pd.MultiIndex.from_product([range(num_penalties), range(num_pdes)], names=["penalty", "pde"])
    columns = [
        "num_agents",
        "top_error",
        "top01_error",
        "mean_error",
        "squared_error",
        "top1_error",
        "top5_error",
        "top10_error",
        "log_num_agents",
        "log_mean_error",
        "log_squared_error",
        "log_top_error",
        "log_top01_error",
        "element_penalty_lambda",
    ]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    # Cast specific columns to the desired dtype
    df["num_agents"] = df["num_agents"].astype(int)
    df["element_penalty_lambda"] = df["element_penalty_lambda"].astype(float)
    return df


def evaluate_mesh_refinement(
    policy_step_function: Callable, environment: AbstractSwarmEnvironment, num_pdes: int
) -> pd.DataFrame:
    """

    Args:
        policy_step_function: A function that takes an observation and returns an action and a value.
        environment: The environment to evaluate.
        num_pdes: The number of pdes to evaluate.

    Returns: A dataframe with the results of the evaluation. Uses multi-indexing for the columns to store the
        num_penalties and num_pdes.

    """
    from modules.swarm_environments.mesh_refinement.evaluation import (
        EvaluationFEMProblemCircularQueue,
    )
    from modules.swarm_environments.mesh_refinement.sweep import SweepMeshRefinement

    fem_config = environment.fem_problem_queue.fem_config
    fem_config["num_pdes"] = num_pdes
    eval_queue = EvaluationFEMProblemCircularQueue(fem_config=fem_config, random_state=np.random.RandomState(seed=123))
    environment.fem_problem_queue = eval_queue
    if isinstance(environment, SweepMeshRefinement):
        environment.train(False)

    num_penalties = 20 if environment.sample_penalty else 1

    # create empty array to store results
    experiment_results = np.empty((num_penalties, num_pdes, 14)) * np.nan

    total_iterations = num_penalties * num_pdes
    with tqdm(total=total_iterations, desc="Evaluating Mesh Refinement") as pbar:
        for current_repeat in range(num_penalties):
            normalized_position = current_repeat / (num_penalties - 1) if num_penalties > 1 else 0.5
            for current_pde in range(num_pdes):
                # the inner loop will cycle through the pdes, so we do not need to take special care during the reset()
                full_additional_information = _single_rollout(
                    policy_step_function=policy_step_function,
                    environment=environment,
                    normalized_position=normalized_position,
                )
                last_additional_information = full_additional_information[-1]
                current_results = get_results_from_additional_information(last_additional_information)
                experiment_results[current_repeat, current_pde, :] = current_results

                pbar.update(1)

    dataframe = _create_dataframe(experiment_results=experiment_results)
    return dataframe
