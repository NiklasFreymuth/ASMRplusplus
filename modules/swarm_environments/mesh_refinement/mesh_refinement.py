from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.basedatatypes import BaseTraceType
from skfem import Mesh
from torch_geometric.data.data import Data

from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.swarm_environments.mesh_refinement.fem_problem_circular_queue import (
    FEMProblemCircularQueue,
)
from modules.swarm_environments.mesh_refinement.fem_problem_wrapper import FEMProblemWrapper
from modules.swarm_environments.mesh_refinement.mesh_refinement_util import (
    convert_error,
    get_aggregation_per_element,
    get_simplex_volumes_from_indices,
    sample_in_range,
)
from modules.swarm_environments.mesh_refinement.visualization.mesh_refinement_visualization import (
    get_plotly_mesh_traces_and_layout,
)
from modules.swarm_environments.util import keys as Keys
from modules.swarm_environments.util.utility_functions import (
    get_default_edge_relation,
    save_concatenate,
)


class MeshRefinement(AbstractSwarmEnvironment):
    """
    Gym environment of a graph-based 2d mesh refinement task using scikit-FEM as a backend.
    Given some underlying finite element task and a coarse initial mesh, the agents must in each step select a number
     of elements of the mesh that will be refined.
     Optionally, a smoothing operator is called after the refinement.
    The goal of the task is to automatically create meshes of adaptive resolution that focus on the "interesting"
    areas of the task with only as many resources as necessary. For this, the mesh is used for a simulation in every
    timestep, and the result of this simulation is compared to some ground truth. The simulation error is then
    combined with a cost for the size of the mesh to produce a reward.
    Intuitively, the task is to highlight important/interesting nodes of the graph, such that the subsequent remeshing
    algorithm assigns more resources to this area than to the other areas. This is iterated over, giving the agent
    the option to iteratively refine the remeshing.
    """

    def __init__(self, environment_config: Dict[Union[str, int], Any], seed: Optional[int] = None):
        """
        Args:
            environment_config: Config for the environment.
                Details can be found in the configs/references/mesh_refinement_reference.yaml example file
            seed: Optional seed for the random number generator.
        """
        super().__init__(environment_config=environment_config, seed=seed)

        self.fem_problem_queue = FEMProblemCircularQueue(
            fem_config=environment_config.get("fem"),
            random_state=np.random.RandomState(seed=seed),
        )
        self.fem_problem: Optional[FEMProblemWrapper] = None

        ################################################
        #        general environment parameters        #
        ################################################
        self._refinement_strategy: str = environment_config.get("refinement_strategy")
        self._max_timesteps = environment_config.get("num_timesteps")
        self._element_limit_penalty = environment_config.get("element_limit_penalty")
        self._maximum_elements = environment_config.get("maximum_elements")
        self._element_penalty_config = self._environment_config.get("element_penalty")
        self._sample_penalty = self._element_penalty_config.get("sample_penalty")

        ################################################
        # graph connectivity, feature and action space #
        ################################################
        self._reward_type = environment_config.get("reward_type")
        self._half_half_reward = environment_config.get("half_half_reward")  # relevant for "spatial" rewards
        self._include_vertices = environment_config.get("include_vertices")
        self._set_graph_sizes()

        ################################################
        #          internal state and cache            #
        ################################################
        self._timestep: int = 0
        self._element_penalty_lambda = None  # 0  # set default value
        self._initial_approximation_errors: Optional[Dict[str, float]] = None
        self._reward = None
        self._cumulative_return: np.array = 0  # return of the environment
        # dictionary containing the error estimation for the current solution for different error evaluation metrics
        self._error_estimation_dict: Optional[Dict[str, np.array]] = None
        self._initial_error_norm = None

        # last-step history for delta-based rewards and plotting
        self._previous_error_per_element: Optional[np.array] = None
        self._previous_num_elements: Optional[int] = None
        self._previous_agent_mapping = None
        self._previous_element_volumes = None
        self._previous_std_per_element = None

        # fields/internal variables for spatial mesh refinement, especially a spatial reward
        self._agent_mapping = None  # mapping List[old_element_indices] of size new_element_indices that maps
        self._reward_per_agent: Optional[np.array] = 0  # cumulative return of the environment per agent
        self._cumulative_reward_per_agent: Optional[np.array] = 0  # cumulative reward of the environment per agent

        # additional policy information that is not passed through the graph
        self._include_additional_policy_information = environment_config.get("include_additional_policy_information")

        self._manual_normalization = environment_config.get(
            "manual_normalization", None
        )  # manually normalize the error
        ################################################
        #            recording and plotting            #
        ################################################
        self._initial_num_elements = None

    def _set_graph_sizes(self):
        """
        Internally sets the
        * action dimension
        * number of node types and node features for each type
        * number of edge types and edge features for each type
        depending on the configuration. Uses the same edge features for all edge types.
        Returns:

        """
        edge_feature_config = self._environment_config.get("edge_features")
        self._edge_features = [
            feature_name for feature_name, include_feature in edge_feature_config.items() if include_feature
        ]
        # set number of edge features
        num_edge_features = 0
        if "distance_vector" in self._edge_features:
            num_edge_features += self.mesh_dimension
        if "euclidean_distance" in self._edge_features:
            num_edge_features += 1

        self._element_feature_functions = self._register_element_features()

        self._num_node_features = len(self._element_feature_functions)
        self._num_node_features += self.fem_problem_queue.num_pde_element_features
        self._num_edge_features = num_edge_features


    def _register_element_features(self) -> Dict[str, Callable[[], np.array]]:
        """
        Returns a dictionary of functions that return the features of the elements. We return a dictionary of functions
        instead of a dictionary of values to allow for lazy evaluation of the features (e.g. if the features are
        expensive to compute or change between iterations, we always want to compute them when requested to).
        Returns:

        """
        element_feature_config = self._environment_config.get("element_features")
        element_feature_names = [
            feature_name for feature_name, include_feature in element_feature_config.items() if include_feature
        ]
        element_features = {}

        if "x_position" in element_feature_names:
            element_features["x_position"] = lambda: self._element_midpoints[:, 0]
        if "y_position" in element_feature_names:
            element_features["y_position"] = lambda: self._element_midpoints[:, 1]
        if "error" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_error"] = lambda i_=position: self.error_per_element[:, i_]
            # we use i_ = position to make sure that the lambda function is evaluated at the correct position
            # in the loop. Otherwise, all lambdas would return the last value of position.
        if "volume" in element_feature_names:
            element_features["volume"] = lambda: self.element_volumes
        if "solution_std" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution_std"] = lambda i_=position: self._solution_std_per_element[:, i_]
        if "solution_mean" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution_mean"] = lambda i_=position: get_aggregation_per_element(
                    self.solution[:, i_],
                    self._element_indices,
                    aggregation_function_str="mean",
                )
        if "solution_min" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution_min"] = lambda i_=position: get_aggregation_per_element(
                    self.solution[:, i_],
                    self._element_indices,
                    aggregation_function_str="min",
                )
        if "solution_max" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution_max"] = lambda i_=position: get_aggregation_per_element(
                    self.solution[:, i_],
                    self._element_indices,
                    aggregation_function_str="max",
                )
        if "solution_median" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution_median"] = lambda i_=position: get_aggregation_per_element(
                    self.solution[:, i_],
                    self._element_indices,
                    aggregation_function_str="median",
                )
        if "element_penalty" in element_feature_names:
            element_features["element_penalty"] = lambda: np.repeat(self._element_penalty_lambda, self._num_elements)
        if "timestep" in element_feature_names:
            element_features["timestep"] = lambda: np.repeat(self._timestep, self._num_elements)
        if "num_elements" in element_feature_names:
            element_features["num_elements"] = lambda: np.repeat(self._num_elements, self._num_elements)
        return element_features


    def reset(self) -> Data:
        """
        Resets the environment and returns an (initial) observation of the next rollout
        according to the reset environment state

        Returns:
            The observation of the initial state.
        """
        # get the next fem problem. This samples a new domain and new load function, resets the mesh and the solution.
        self.fem_problem = self.fem_problem_queue.next()

        # calculate the solution of the finite element problem for the initial mesh and retrieve an error per element
        self._error_estimation_dict = self.fem_problem.calculate_solution_and_get_error()

        # reset the internal state of the environment. This includes the current timestep, the current element penalty
        # and some values for calculating the reward and plotting the env
        self._reset_internal_state()

        observation = self.last_observation
        return observation

    def _reset_internal_state(self):
        """
        Resets the internal state of the environment
        Returns:

        """
        self._agent_mapping = np.arange(self._num_elements).astype(np.int64)  # map to identity at first step
        self._previous_agent_mapping = np.arange(self._num_elements).astype(np.int64)  # map to identity at first step
        self._previous_element_volumes = self.element_volumes
        self._previous_std_per_element = self._solution_std_per_element
        self._reward_per_agent = np.zeros(self.num_agents)
        self._cumulative_reward_per_agent = np.zeros(self._num_elements)

        # reset timestep and rewards
        self._timestep = 0
        self._reward = 0
        self._cumulative_return = 0

        # reset internal state that tracks statistics over the episode
        self._previous_error_per_element = self.error_per_element

        # collect a dictionary of initial errors to normalize them when calculating metrics during evaluation
        self._initial_approximation_errors = self._calculate_initial_approximation_errors()

        self._previous_num_elements = self._num_elements
        self._initial_num_elements = self._num_elements

        if self.error_per_element is not None:
            self._initial_error_norm = np.linalg.norm(self.error_per_element, axis=0)

        # reset the element penalty, necessary if it is sampled
        if self._sample_penalty:
            sampling_type = self._element_penalty_config.get("sampling_type", "loguniform")
            min_value = self._element_penalty_config.get("min")
            max_value = self._element_penalty_config.get("max")
            element_penalty_lambda = sample_in_range(max_value, min_value, sampling_type)
            self._element_penalty_lambda = element_penalty_lambda
        else:  # element penalty is a scalar value
            self._element_penalty_lambda = self._element_penalty_config.get("value")

    def _calculate_initial_approximation_errors(self):
        if self._manual_normalization:
            return {error_name: self._manual_normalization for error_name in self.error_estimation_dict}
        else:
            return {
                error_name: (
                    np.max(np.atleast_2d(errors), axis=0)
                    if "maximum" in error_name
                    else np.sum(np.atleast_2d(errors), axis=0)
                )
                + 1.0e-12
                for error_name, errors in self.error_estimation_dict.items()
            }

    def step(self, action: np.ndarray) -> Tuple[Data, np.array, bool, Dict[str, Any]]:
        """
        Performs a step of the Mesh Refinement task
        Args:
            action: the action the agents will take in this step. Has shape (num_agents, action_dimension)
            Given as an array of shape (num_agents, action_dimension)

        Returns: A 4-tuple (observations, reward, done, info), where
            * observations is a graph of the agents and their positions, in this case of the refined mesh
            * reward is a single scalar shared between all agents, i.e., per **graph**
            * done is a boolean flag that says whether the current rollout is finished or not
            * info is a dictionary containing additional information
        """
        assert not self.is_terminal, (
            f"Tried to perform a step on a terminated environment. Currently on step "
            f"{self._timestep:} of {self._max_timesteps:} "
            f"with {self._num_elements}/{self._maximum_elements} elements."
        )

        self._timestep += 1

        self._set_previous_step()

        # refine mesh and store which element has become which set of new elements
        self._agent_mapping = self._refine_mesh(action=action)

        # solve equation and calculate error per element/element
        self._previous_error_per_element = self.error_per_element

        self._error_estimation_dict = self.fem_problem.calculate_solution_and_get_error()

        # query returns
        observation = self.last_observation

        reward_dict = self._get_reward_dict()
        metric_dict = self._get_metric_dict()
        action_dict = self._get_action_dict(action=action)

        # done after a given number of steps or if the mesh becomes too large
        done = self.is_terminal
        info = (
            reward_dict
            | metric_dict
            | action_dict
            | {
                Keys.IS_TRUNCATED: self.is_truncated,
                Keys.RETURN: self._cumulative_return,
            }
        )
        return observation, self._reward, done, info

    def inference_step(self, action: np.ndarray) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        Performs a step of the Mesh Refinement task *without* calculating the reward or difference to the fine-grained
        reference. This is used for inference

        Args:
            action: the action the agents will take in this step. Has shape (num_agents, action_dimension)
            Given as an array of shape (num_agents, action_dimension)

        Returns: A 4-tuple (observations, reward, done, info), where
            * observations is a graph of the agents and their positions, in this case of the refined mesh
            * reward is a single scalar shared between all agents, i.e., per **graph**
            * done is a boolean flag that says whether the current rollout is finished or not
            * info is a dictionary containing additional information
        """
        assert not self.is_terminal, (
            f"Tried to perform a step on a terminated environment. Currently on step "
            f"{self._timestep:} of {self._max_timesteps:} "
            f"with {self._num_elements}/{self._maximum_elements} elements."
        )

        self._timestep += 1
        self._agent_mapping = self._refine_mesh(action=action)
        # solve equation
        self.fem_problem.calculate_solution()
        observation = self.last_observation
        done = self.is_terminal
        info = {}
        return observation, self._reward, done, info

    def _set_previous_step(self):
        """
        Sets variables for the previous timestep. These are used for the reward function, as well as for different
        kinds of plots and metrics
        """
        self._previous_num_elements = self._num_elements
        self._previous_agent_mapping = self._agent_mapping
        self._previous_element_volumes = self.element_volumes
        self._previous_std_per_element = self._solution_std_per_element

    def _refine_mesh(self, action: np.array) -> np.array:
        """
        Refines fem_problem.mesh by splitting all faces/elements for which the average of agent activation surpasses a
        threshold.
        If this refinement exceeds the maximum number of nodes allowed in the environment, we return a boolean flag
        that indicates so and stops the environment

        Optionally smoothens the newly created mesh as a post-processing step

        Args:
            action: An action/activation per element.
                Currently, a scalar value that is interpreted as a refinement threshold.
        Returns: An array of mapped element indices

        """
        # make sure that action is 1d, i.e., decides on a refinement threshold per agent
        action = action.flatten()

        elements_to_refine = self._get_elements_to_refine(action)

        # updates self.fem_problem.mesh
        element_mapping = self.fem_problem.refine_mesh(elements_to_refine)
        return element_mapping

    def _get_elements_to_refine(self, action: np.array) -> np.array:
        """
        Calculate which elements to refine based on the action, refinement strategy and the
        maximum number of elements allowed in the environment
        Args:
            action: An action/activation per agent, i.e., per element or element

        Returns: An array of ids corresponding to elements_to_refine

        """
        # select elements to refine based on the average actions of its surrounding agents/nodes

        # refine elements w/ action > 0
        if self._refinement_strategy in ["absolute", "absolute_discrete"]:
            elements_to_refine = np.argwhere(action > 0.0).flatten()
        elif self._refinement_strategy in ["argmax", "single_agent"]:
            if action.size == 1 and isinstance(action.item(), int):
                elements_to_refine = action
                if self._refinements_per_element[elements_to_refine] > 30:
                    # if we have refined the selected element too often, we skip this action
                    elements_to_refine = np.array([])
            else:  # action is a vector of size num_agents containing scores for each element
                action[self._refinements_per_element > 30] = -np.inf
                elements_to_refine = np.argmax(action).flatten()
        else:
            raise ValueError(f"Unknown refinement strategy '{self._refinement_strategy}")
        return elements_to_refine

    def render(
        self,
        mode: str = "human",
        render_intermediate_steps: bool = False,
        *args,
        **kwargs,
    ) -> Union[np.array, Tuple[List[BaseTraceType], Dict[str, Any]]]:
        """
        Renders and returns a list of plotly traces  for the *current state* of this environment.

        Args:
            mode: How to render the figure. Must either be "human" or "rgb_array"
            render_intermediate_steps: Whether to render steps that are non-terminal or not. Defaults to False, as
              we usually want to visualize the result of a full rollout
            *args: additional arguments (not used).
            **kwargs: additional keyword arguments (not used).

        Returns: Either a numpy array of the current figure if mode=="rgb_array", or a tuple
           (traces, additional_render_information) otherwise.
        """
        if render_intermediate_steps or self.is_terminal:
            # only return traces if told to do so or if at the last step to avoid overlay of multiple steps
            remaining_error = self._get_remaining_error(return_dimensions=False)
            title = (
                f"Solution. "
                f"Element Penalty: {self._element_penalty_lambda:.1e} "
                f"Reward: {np.sum(self._reward):.3f} "
                f"Return: {np.sum(self._cumulative_return):.3f} "
                f"Agents: {self.num_agents} "
                f"Remaining Error: {remaining_error:.3f}"
            )

            traces, layout = get_plotly_mesh_traces_and_layout(
                mesh=self.mesh,
                scalars=self.scalar_solution,
                mesh_dimension=self.mesh_dimension,
                title=title,
                boundary=self.fem_problem.plot_boundary,
            )
            return traces, {"layout": layout}
        else:
            return [], {}

    def _get_remaining_error(self, return_dimensions: bool = False) -> Union[np.array, Tuple]:
        """
        Get the remaining error by aggregating over all elements and taking the convex sum of all solution dimensions
        """
        if "maximum" in self.error_metric:
            remaining_error_per_dimension = np.max(self.error_per_element, axis=0)
        else:
            remaining_error_per_dimension = np.sum(self.error_per_element, axis=0)

        remaining_error_per_dimension = remaining_error_per_dimension / self.initial_approximation_error  # normalize
        remaining_error = self.project_to_scalar(remaining_error_per_dimension)

        if return_dimensions:
            return remaining_error, remaining_error_per_dimension
        else:
            return remaining_error

    @property
    def last_observation(self) -> Data:
        """
        Retrieve an observation graph for the current state of the environment.

        We use an additional self.last_observation wrapper to make sure that classes that inherit from this
        one have access to node and edge features outside the Data() structure
        Returns: A Data() object of the graph that describes the current state of this environment

        """
        graph_dict = {}
        graph_dict = graph_dict | self._get_graph_nodes()
        graph_dict = graph_dict | self._get_graph_edges()

        observation_graph = Data(**graph_dict)

        return observation_graph

    def _get_graph_nodes(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns a dictionary of node features that are used to describe the current state of this environment.

        Returns: A dictionary of node features. This dictionary has the format
        {"x": element_features}
        where element and node features depend on the context, but include things like the evaluation of the target
        function, the degree of the node, etc.
        """
        # Builds feature matrix of shape (#elements, #features)
        # by iterating over the functions in self._element_feature_functions.
        element_features = np.array([fn() for key, fn in self._element_feature_functions.items()]).T
        element_features = save_concatenate([element_features, self.fem_problem.element_features()], axis=1)
        element_features = torch.tensor(element_features, dtype=torch.float32)
        node_dict = {"x": element_features}
        return node_dict

    def _get_graph_edges(
        self,
    ) -> Dict[Union[str, Tuple[str, str, str]], Dict[str, torch.Tensor]]:
        """
        Returns a dictionary of edge features that are used to describe the current state of this environment.
        Note that we always use symmetric graphs and self edges.

        Returns: A dictionary of edge features. This dictionary has the format
        {
        "edge_index": indices,
        "edge_attr": features
        }
        """
        edge_index, edge_attr = self._element2element_features(self._num_edge_features)
        edge_dict = {"edge_index": edge_index, "edge_attr": edge_attr}
        return edge_dict

    def _element2element_features(self, num_edge_features: int):
        # concatenate incoming, outgoing and self edges of each node to get an undirected graph
        src_nodes = np.concatenate(
            (
                self._element_neighbors[0],
                self._element_neighbors[1],
                np.arange(self._num_elements),
            ),
            axis=0,
        )
        dest_nodes = np.concatenate(
            (
                self._element_neighbors[1],
                self._element_neighbors[0],
                np.arange(self._num_elements),
            ),
            axis=0,
        )
        num_edges = self._element_neighbors.shape[1] * 2 + self._num_elements
        edge_features = np.empty(shape=(num_edges, num_edge_features))
        edge_feature_position = 0
        if "distance_vector" in self._edge_features:
            distance_vectors = self._element_midpoints[dest_nodes] - self._element_midpoints[src_nodes]
            edge_features[:, edge_feature_position : edge_feature_position + self.mesh_dimension] = distance_vectors
            edge_feature_position += self.mesh_dimension
        if "euclidean_distance" in self._edge_features:
            euclidean_distances = np.linalg.norm(
                self._element_midpoints[dest_nodes] - self._element_midpoints[src_nodes],
                axis=1,
            )
            edge_features[:, edge_feature_position] = euclidean_distances
            edge_feature_position += 1
        edge_index = torch.tensor(np.vstack((src_nodes, dest_nodes))).long()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        return edge_index, edge_attr

    def _get_reward_dict(self) -> Dict[str, np.float32]:
        """
        Calculate the reward for the current timestep depending on the environment states and the action
        the agents took.
        Args:

        Returns:
            Dictionary that must contain "reward" as well as partial reward data

        """
        reward, reward_dict = self._get_reward_by_type()

        self._reward = reward
        self._cumulative_return = self._cumulative_return + np.sum(self._reward)

        return reward_dict

    def _get_metric_dict(self) -> Dict[str, Any]:
        remaining_error, remaining_error_per_dimension = self._get_remaining_error(return_dimensions=True)

        metric_dict = {
            Keys.REMAINING_ERROR: remaining_error,
            Keys.ERROR_TIMES_AGENTS: remaining_error * self.num_agents,
            Keys.DELTA_ELEMENTS: self._num_elements - self._previous_num_elements,
            Keys.AVG_TOTAL_REFINEMENTS: np.log(self._num_elements / self._initial_num_elements) / np.log(4),
            Keys.AVG_STEP_REFINEMENTS: np.log(self._num_elements / self._previous_num_elements) / np.log(4),
            Keys.NUM_ELEMENTS: self._num_elements,
            Keys.NUM_AGENTS: self.num_agents,
            Keys.REACHED_ELEMENT_LIMITS: self.reached_element_limits,
            Keys.REFINEMENT_STD: self._refinements_per_element.std(),
        }
        if 1 < len(self._solution_dimension_names) < 5:
            # provide individual metrics for the different solution dimensions
            for position, name in enumerate(self._solution_dimension_names):
                metric_dict[f"{name}_remaining_error"] = remaining_error_per_dimension[position]
        elif len(self._solution_dimension_names) >= 5:
            # provide a single metric that is the weighted sum of the different solution dimensions
            last_dimension = self._solution_dimension_names[-1]
            metric_dict[f"{last_dimension}_remaining_error"] = remaining_error_per_dimension[-1]

        for error_metric, element_errors in self.error_estimation_dict.items():
            if element_errors.shape[0] == self._num_elements:
                # need to aggregate
                if "maximum" in error_metric:
                    error_per_dimension = np.max(element_errors, axis=0)
                else:
                    error_per_dimension = np.sum(element_errors, axis=0)

            else:
                error_per_dimension = element_errors
            error_per_dimension = error_per_dimension / self._initial_approximation_errors[error_metric]
            remaining_error = self.project_to_scalar(error_per_dimension)
            metric_dict[f"{error_metric}_error"] = remaining_error
        return metric_dict

    def _get_action_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dictionary of information about the action that was taken in the current timestep
        Args:
            action: The action that was taken in the current timestep

        Returns: A dictionary of information about the action that was taken in the current timestep

        """
        action_dict = {}
        if self._refinement_strategy in ["absolute", "absolute_discrete"]:
            action_dict["action_mean"] = np.mean(action)
            action_dict["action_std"] = np.std(action)
        return action_dict

    def _get_reward_by_type(self) -> Tuple[np.array, Dict]:
        """
        Calculate the reward for the current timestep depending on the environment states and the action
        the agents took.

        Args:

        Returns: A Tuple (reward, dictionary of partial rewards).

        """

        reward_dict = {}

        if self._reward_type in ["spatial", "spatial_volume", "spatial_max"]:
            reward_per_agent_and_dim = np.copy(self._previous_error_per_element)

            if self._reward_type == "spatial_max":
                # calculate the difference between an elements error and the maximum error of its childs.
                # Only makes sense if we have the "maximum" as an error metric per element.
                assert self.error_metric == "maximum", (
                    f"Must use maximum error metric for spatial_max reward type" f"but got {self.error_metric}"
                )

                # use torch here because numpy does not have a scatter_max function
                from torch_scatter import scatter_max

                agent_mapping = torch.tensor(self.agent_mapping)
                error_per_element = torch.tensor(self.error_per_element)
                max_mapped_error, _ = scatter_max(
                    src=error_per_element,
                    index=agent_mapping,
                    dim=0,
                    dim_size=reward_per_agent_and_dim.shape[0],
                )
                max_mapped_error = max_mapped_error.numpy()
                reward_per_agent_and_dim = reward_per_agent_and_dim - max_mapped_error

            elif self._reward_type in ["spatial", "spatial_volume"]:
                np.add.at(
                    reward_per_agent_and_dim,
                    self.agent_mapping,
                    -self.error_per_element,
                )  # scatter add

                # normalize reward per element by its inverse volume. I.e., give more weight to smaller volumes
                if self._reward_type == "spatial_volume":
                    reward_per_agent_and_dim = reward_per_agent_and_dim / self._previous_element_volumes[:, None]

                # We could also normalize by the total mesh volume here, but that is redundant
                # with the initial error normalization below
                # In the normalized variant, also make sure that the total volume of the mesh is considered
            else:
                raise ValueError(f"Unknown reward type {self._reward_type}")

            reward_per_agent_and_dim = reward_per_agent_and_dim / self.initial_approximation_error  # normalize to 1

            reward_per_agent = self.project_to_scalar(reward_per_agent_and_dim)

            # apply action/element penalty
            # the element penalty per agent depends on how many new elements are created by this agent

            element_counts = np.unique(self.agent_mapping, return_counts=True)[1]
            element_counts = element_counts - 1  # -1 because of the original element
            element_penalty = self._element_penalty_lambda * element_counts
            element_limit_penalty = (
                (self._element_limit_penalty / self._previous_num_elements) if self.reached_element_limits else 0
            )
            reward_per_agent = reward_per_agent - element_penalty - element_limit_penalty

            # potentially transform reward
            if self._half_half_reward:
                reward_per_agent = (reward_per_agent + np.mean(reward_per_agent)) / 2

            self._reward_per_agent = reward_per_agent
            self._cumulative_reward_per_agent = (
                self._cumulative_reward_per_agent[self._previous_agent_mapping] + reward_per_agent
            )
            reward = reward_per_agent

        elif self._reward_type == "vdgn":
            # this uses the multi-objective VDGN reward function of https://arxiv.org/pdf/2211.00801v3.pdf,
            # which is a log difference of the error per dimension.
            # The element penalty in the paper is defined as ((d_t-1 - d_t)/d_thresh)*w,
            # i.e., as the difference in elements divided by a threshold.
            # This equals self._element_penalty_lambda * (self.num_elements - self._previous_num_elements)
            # since the (1/d_thresh)*w is a linear factor
            previous_errors = np.linalg.norm(self._previous_error_per_element, axis=0)
            current_errors = np.linalg.norm(self.error_per_element, axis=0)
            reward_per_dimension = np.log(previous_errors + 1.0e-12) - np.log(current_errors + 1.0e-12)
            # 1.0e-12 for stability reasons, as the error may become 0 for coarse reference solutions

            reward = self.project_to_scalar(reward_per_dimension)  # scalar reward. Will broadcast to all agents
            # according to the VDGN value decomposition

            element_penalty = self._element_penalty_lambda * (self._num_elements - self._previous_num_elements)
            element_limit_penalty = self._element_limit_penalty if self.reached_element_limits else 0

            reward = reward - element_penalty - element_limit_penalty

        elif self._reward_type in ["argmax", "single_agent"]:
            previous_errors = np.linalg.norm(self._previous_error_per_element, axis=0)
            current_errors = np.linalg.norm(self.error_per_element, axis=0)
            reward_per_dimension = (previous_errors - current_errors) / self._initial_error_norm
            reward = self.project_to_scalar(reward_per_dimension)
            element_penalty = 0
            element_limit_penalty = 0
        else:
            raise ValueError(f"Unknown reward type '{self._reward_type}'")

        reward_dict[Keys.REWARD] = reward
        reward_dict[Keys.PENALTY] = -reward
        reward_dict[Keys.ELEMENT_LIMIT_PENALTY] = element_limit_penalty
        reward_dict[Keys.ELEMENT_PENALTY] = element_penalty
        reward_dict[Keys.ELEMENT_PENALTY_LAMBDA] = self._element_penalty_lambda
        return reward, reward_dict

    @property
    def mesh(self) -> Mesh:
        """
        Returns the current mesh.
        """
        return self.fem_problem.mesh

    @property
    def agent_node_type(self) -> str:
        return Keys.ELEMENT

    @property
    def _vertex_positions(self) -> np.array:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)
        """
        return self.fem_problem.vertex_positions

    @property
    def _element_indices(self) -> np.array:
        return self.fem_problem.element_indices

    @property
    def _element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements/faces.
        Returns: np.array of shape (num_elements, 2)

        """
        return self.fem_problem.element_midpoints

    @property
    def _mesh_edges(self) -> np.array:
        """
        Returns: the edges of all vertices/nodes of the mesh. Shape (2, num_edges)
        """
        return self.fem_problem.mesh_edges

    @property
    def _element_neighbors(self) -> np.array:
        """
        Find neighbors of each element. Shape (2, num_neighbors)
        Returns:

        """
        # f2t are element/face neighborhoods, which are set to -1 for boundaries
        return self.fem_problem.element_neighbors

    @property
    def _num_elements(self) -> int:
        return len(self._element_indices)

    @property
    def _num_vertices(self) -> int:
        return len(self._vertex_positions)

    @property
    def element_volumes(self) -> np.array:
        return get_simplex_volumes_from_indices(positions=self._vertex_positions, simplex_indices=self._element_indices)

    @property
    def num_node_features(self) -> int:
        return self._num_node_features

    @property
    def num_edge_features(self) -> int:
        return self._num_edge_features

    @property
    def action_dimension(self) -> int:
        """
        Returns: The dimensionality of the action space. Note that for discrete action spaces
        (i.e., refinement_strategy == "absolute_discrete")  this is the number of possible actions,
        which are all represented in a single scalar

        """
        if self._refinement_strategy == "absolute_discrete":
            return 2
        else:  # single continuous value
            return 1

    @property
    def num_agents(self) -> int:
        if self.fem_problem is not None and self.fem_problem.mesh is not None:
            return self._num_elements
        else:
            return 1  # placeholder

    @property
    def _action_space(self) -> gym.Space:
        """

        Returns: The **current** action space of the environment. Bound to change, since the number of agents
        changes

        """
        if self._refinement_strategy in ["absolute_discrete", "argmax", "single_agent"]:
            return gym.spaces.MultiDiscrete([self.action_dimension] * self.num_agents)
        else:
            return gym.spaces.Box(
                low=-1e5,
                high=1e5,
                shape=(
                    self.num_agents,
                    self.action_dimension,
                ),
                dtype=np.float32,
            )

    @property
    def agent_mapping(self) -> np.array:
        assert self._agent_mapping is not None, "Element mapping not initialized"
        return self._agent_mapping

    @property
    def previous_agent_mapping(self) -> np.array:
        assert self._previous_agent_mapping is not None, "Previous element mapping not initialized"
        return self._previous_agent_mapping

    @property
    def reached_element_limits(self) -> bool:
        """
        True if the number of elements/faces in the mesh is above the maximum allowed value.
        Returns:

        """
        return self._num_elements > self._maximum_elements

    @property
    def is_truncated(self) -> bool:
        return self._timestep >= self._max_timesteps

    @property
    def is_terminal(self) -> bool:
        return self.reached_element_limits or self.is_truncated

    @property
    def solution(self) -> np.array:
        """
        Returns: solution vector per *vertex* of the mesh.
            An array (num_vertices, solution_dimension),
            where every entry corresponds to the solution of the parameterized fem_problem
            equation at the position of the respective node/vertex.

        """
        return self.fem_problem.nodal_solution

    @property
    def _solution_dimension_names(self) -> List[str]:
        """
        Plotting utility that gives the names of the solution dimensions.
        Returns:

        """
        return self.fem_problem_queue.solution_dimension_names

    def project_to_scalar(self, values: np.array) -> np.array:
        """
        Projects a value per node or graph and solution dimension to a scalar value per node.
        Args:
            values: A vector of shape ([num_vertices/nodes,] solution_dimension)

        Returns: A scalar value per vertex
        """
        return self.fem_problem.project_to_scalar(values)

    @property
    def scalar_solution(self):
        return self.project_to_scalar(self.solution)

    @property
    def error_metric(self) -> str:
        """
        'Main' error estimation method used for the algorithm.
        This is the method that is used for the reward calculation.
        Other methods can be used for debugging and logging purposes.
        Returns:

        """
        return self.fem_problem.error_metric

    @property
    def error_per_element(self) -> np.array:
        """
        Returns: error per element of the mesh. np.array of shape (num_elements, solution_dimension)

        """
        return self._error_estimation_dict.get(self.error_metric)

    @property
    def initial_approximation_error(self) -> np.array:
        """
        Returns: error per element of the mesh. np.array of shape (num_elements, solution_dimension)

        """
        return self._initial_approximation_errors.get(self.error_metric)

    @property
    def error_estimation_dict(self) -> Dict[str, np.array]:
        """
        Returns a dictionary of all error estimation methods and their respective errors.
        These errors may be per element/face, or per integration point, depending on the metric.
        Returns:

        """
        return self._error_estimation_dict

    @property
    def _refinements_per_element(self) -> np.array:
        return self.fem_problem.refinements_per_element

    @property
    def _solution_std_per_element(self) -> np.array:
        """
        Computes the standard deviation of the solution per element.
        Returns: np.array of shape (num_elements, solution_dimension)

        """
        return get_aggregation_per_element(self.solution, self._element_indices, aggregation_function_str="std")

    @property
    def sample_penalty(self) -> bool:
        return self._sample_penalty

    @property
    def refinement_strategy(self) -> str:
        return self._refinement_strategy

    @property
    def has_homogeneous_graph(self) -> bool:
        return not self._include_vertices

    @property
    def mesh_dimension(self) -> int:
        return self.fem_problem_queue.mesh_dimension

    def set_element_penalty_lambda(self, position_or_value: float, from_position: bool = True):
        """
        Sets the element penalty lambda from the provided position.
        Args:
            position_or_value: A float between 0 and 1 that determines the element penalty lambda if from_position.
                Otherwise, the value of the element penalty lambda.
            from_position: If True, the element penalty lambda is taken log-uniformly from the provided position,
                regardless of how the value is usually sampled.


        Returns: None

        Note: Sets self._element_penalty_lambda

        """
        element_penalty_config = self._environment_config.get("element_penalty")

        if element_penalty_config.get("sample_penalty"):
            if from_position:
                # sample element penalty loguniformly for comparison between methods
                log_min = np.log(element_penalty_config.get("min"))
                log_max = np.log(element_penalty_config.get("max"))
                self._element_penalty_lambda = np.exp(position_or_value * log_min + (1 - position_or_value) * log_max)
            else:  # fixed element penalty
                self._element_penalty_lambda = position_or_value
        else:
            # element penalty is fixed
            self._element_penalty_lambda = element_penalty_config.get("value")

    ####################
    # additional plots #
    ####################

    def _plot_value_per_element(
        self,
        value_per_element: np.array,
        title: str,
        normalize_by_element_volume: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> go.Figure:
        """
        only return traces if asked or at the last step to avoid overlay of multiple steps
        Args:
            value_per_element: A numpy array of shape (num_elements,).
            title: The title of the plot.
            normalize_by_element_volume: If True, the values are normalized by the element volume as value /= element_volume.
            mesh: The mesh to plot the values on. If None, the mesh of the current state is used.
        Returns: A plotly figure with an outline of the mesh and value per element in the element midpoints.

        """
        if mesh is None:
            assert len(value_per_element) == self.num_agents, (
                f"Need to provide a value per agent, given " f"'{value_per_element.shape}' and '{self.num_agents}'"
            )
            mesh = self.fem_problem.mesh
            mesh_dimension = self.mesh_dimension
        else:
            mesh_dimension = mesh.dim()

        if normalize_by_element_volume:
            value_per_element = value_per_element / self.element_volumes

        boundary = self.fem_problem.plot_boundary
        traces, layout = get_plotly_mesh_traces_and_layout(
            mesh=mesh,
            scalars=value_per_element,
            mesh_dimension=mesh_dimension,
            title=title,
            boundary=boundary,
        )

        value_per_element_plot = go.Figure(data=traces, layout=layout)
        return value_per_element_plot

    def _plot_error_per_element(self, normalize_by_element_volume: bool = True) -> go.Figure:
        weighted_remaining_error = self._get_remaining_error(return_dimensions=False)
        return self._plot_value_per_element(
            value_per_element=self.project_to_scalar(self.error_per_element),
            normalize_by_element_volume=normalize_by_element_volume,
            title=f"Element Errors. Remaining total error: {weighted_remaining_error:.4f}",
        )

    def additional_plots(self, iteration: int, policy_step_function: Optional[callable] = None) -> Dict[str, go.Figure]:
        """
        Function that takes an algorithm iteration as input and returns a number of additional plots about the
        current environment as output. Some plots may be always selected, some only on e.g., iteration 0.
        Args:
            iteration: The current iteration of the algorithm.
            policy_step_function: (Optional)
                A function that takes a graph as input and returns the action(s) and (q)-value(s)
                for each agent.

        """
        _, remaining_error_per_solution_dimension = self._get_remaining_error(return_dimensions=True)
        additional_plots = {
            "refinements_per_element": self._plot_value_per_element(
                value_per_element=self._refinements_per_element,
                title="Refinements per element",
            ),
            "scalar_solution_std_per_element": self._plot_value_per_element(
                value_per_element=self.project_to_scalar(self._solution_std_per_element),
                title=f"Element Std of Solution Norm",
            ),
            "scalar_solution_error_per_element": self._plot_error_per_element(normalize_by_element_volume=False),
        }

        if len(self._solution_dimension_names) > 1:
            # plot individual dimensions of multi-dimensional objective
            subsampled_solution_dimensions = self._solution_dimension_names
            if len(self._solution_dimension_names) > 5:
                subsampled_solution_dimensions = [
                    self._solution_dimension_names[0],
                    self._solution_dimension_names[-1],
                ]
            for position, solution_dimension_name in enumerate(subsampled_solution_dimensions):
                additional_plots[f"{solution_dimension_name}_std_per_element"] = self._plot_value_per_element(
                    value_per_element=self._solution_std_per_element[:, position],
                    title=f"Element Std for '{solution_dimension_name}'",
                )

                additional_plots[f"{solution_dimension_name}_error_per_element"] = self._plot_value_per_element(
                    value_per_element=self.error_per_element[:, position],
                    normalize_by_element_volume=False,
                    title=f"Error per element of '{solution_dimension_name}'. "
                    f"Remaining error: {remaining_error_per_solution_dimension[position]:.4f}",
                )

        if policy_step_function is not None:
            from modules.swarm_environments.util.torch_util import detach

            actions, values = policy_step_function(observation=self.last_observation)
            if len(actions) == self._num_elements:
                additional_plots["final_actor_evaluation"] = self._plot_value_per_element(
                    detach(actions),
                    title=f"Action per Agent at step {self._timestep}",
                )
            if len(values) == self._num_elements:
                additional_plots["final_critic_evaluation"] = self._plot_value_per_element(
                    detach(values),
                    title=f"Critic Evaluation at step {self._timestep}",
                )
        if self._reward_type in ["spatial", "spatial_max", "spatial_volume"]:
            additional_plots["cumulative_reward"] = self._plot_value_per_element(
                value_per_element=self._cumulative_reward_per_agent,
                title="Cumulative Reward",
                mesh=self.fem_problem.previous_mesh,
            )
            additional_plots["reward_per_agent"] = self._plot_value_per_element(
                value_per_element=self._reward_per_agent,
                title="Final Reward",
                mesh=self.fem_problem.previous_mesh,
            )
        additional_plots |= self.fem_problem.additional_plots()
        return additional_plots

    def __deepcopy__(self, memo):
        """
        Overwrite deepcopy to reinitialize stateless (lambda-) functions
        it is sufficient to call the register functions,
        as only new objects for the stateless lambda functions have to be created
        Args:
            memo:

        Returns:

        """
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        setattr(result, "_element_feature_functions", result._register_element_features())
        return result
