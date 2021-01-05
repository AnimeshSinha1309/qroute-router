"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing

import numpy as np
import tqdm
import torch

from qroute.environment.state import CircuitStateDQN
from qroute.models.graph_dual import GraphDualModel
from qroute.environment.env import step


class MCTSAgent:
    """
    Monte Carlo Tree Search combiner object for evaluating the combination of moves
    that will form one step of the simulation.
    This at the moment does not look into the future steps, just calls an evaluator
    """

    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 0.01

    class MCTSState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        HYPERPARAM_NOISE_ALPHA = 0.2
        HYPERPARAM_PRIOR_FRACTION = 0.25

        def __init__(self, state, solution=None, r_previous=0, parent=None):
            """
            Initialize a new state
            """
            self.state: CircuitStateDQN = state
            self.parent_state, self.parent_action = parent
            self.r_previous = r_previous
            self.num_actions = len(self.state.device.edges) + 1
            self.solution: np.ndarray = copy.copy(solution) if solution is not None else np.zeros(self.num_actions)

            self.rollout_reward = self.calc_rollout_reward() if self.parent_action is not None else 0.0
            self.action_mask = self.state.device.swappable_edges(self.solution)

            self.n_value = torch.zeros(self.num_actions)
            self.q_value = torch.zeros(self.num_actions)
            self.child_states: typing.List[typing.Optional[MCTSAgent.MCTSState]] = \
                [None for _ in range(self.num_actions)]

            self.model = GraphDualModel(state.device)
            self.priors, _ = self.model.predict_pi(self.solution)

            self.priors[(self.action_mask == 0).nonzero()] = -1e8
            self.priors = (torch.softmax(self.priors, dim=0) * torch.FloatTensor(self.action_mask)).flatten()
            self.prior_noise()

        def update_q(self, reward, index):
            self.q_value[index] = (self.q_value[index] * self.n_value[index] + reward) / (self.n_value[index] + 1)
            self.n_value[index] += 1

        def select(self, c=0.01) -> int:
            """
            Select one of the child actions based on UCT rule
            """
            n_visits = torch.sum(self.n_value).item()
            uct = self.q_value + (self.priors * c * np.sqrt(n_visits + 1) / (self.n_value + 1))
            best_val, _ = torch.max(uct)
            best_moves: torch.Tensor = torch.eq(best_val, uct)
            best_move_indices, _ = torch.where(best_moves)
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def prior_noise(self):
            """
            Adds dirichlet noise to priors.
            Called when the state is the root node
            """
            noise = np.random.dirichlet([self.HYPERPARAM_NOISE_ALPHA] * int(sum(self.solution)),
                                        size=self.solution.shape)
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise

        def calc_rollout_reward(self):
            """
            performs R random rollout, the total reward in each rollout is computed.
            returns: mean across the R random rollouts.
            """
            next_state, _, _, _ = step(self.solution, self.state)
            _, self.rollout_reward = self.model(next_state)
            return self.rollout_reward

    def __init__(self, state):
        self.state = state
        self.root = self.MCTSState(state)
        self.model = GraphDualModel(state.device)

    def search(self, n_mcts, c):
        """Perform the MCTS search from the root"""
        for _ in tqdm.trange(n_mcts):
            mcts_state = self.root  # reset to root for new trace

            while not np.any(self.state.device.swappable_edges(mcts_state.solution)):
                action_index = mcts_state.select(c=c)
                if mcts_state.child_states[action_index] is not None:
                    # MCTS Algorithm: SELECT STAGE
                    mcts_state = mcts_state.child_states[action_index]
                    continue
                else:
                    # MCTS Algorithm: EXPAND STAGE
                    next_solution = np.copy(mcts_state.solution)
                    next_solution[action_index] = True
                    _next_state, reward, _done, _debug = step(next_solution, self.state)
                    mcts_state.child_states[action_index] = self.MCTSState(
                        self.state, next_solution, reward, (self, action_index))
                    mcts_state = mcts_state.child_states[action_index]
                    break

            # MCTS Algorithm: BACKUP STAGE
            total_reward = mcts_state.rollout_reward
            while mcts_state.parent_action is not None:
                total_reward = mcts_state.r_previous + self.HYPERPARAM_DISCOUNT_FACTOR * total_reward
                mcts_state = mcts_state.parent_state
                mcts_state.update_q(total_reward, mcts_state.parent_action)

    def return_results(self, temp):
        """Process the output at the root node"""

        def stable_normalizer(x, temperature):
            """Computes x[i]**temp/sum_i(x[i]**temp)"""
            x = (x / torch.max(x)) ** temperature
            return torch.abs(x / torch.sum(x))

        pi_target = stable_normalizer(self.root.n_value, temp)
        action_1 = np.expand_dims(np.array(pi_target), -1).reshape(20, -1)
        action_1 = np.sum(action_1, 1)
        v_target = torch.sum((self.root.n_value / torch.sum(self.root.n_value)) * self.root.q_value)
        return self.root, pi_target.numpy(), torch.FloatTensor([v_target]), action_1

    def forward(self, a):
        """Move the root forward"""
        self.root = self.root.child_actions[a].child_state
