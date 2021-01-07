"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing as ty

import numpy as np
import torch

from metas import ReplayMemory
from qroute.metas import CombinerAgent
from qroute.environment.state import CircuitStateDQN
from qroute.environment.env import step


class MCTSAgent(CombinerAgent):

    class MCTSState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        HYPERPARAM_NOISE_ALPHA = 0.2
        HYPERPARAM_PRIOR_FRACTION = 0.25

        def __init__(self, state, model, solution=None, r_previous=0, parent_state=None, parent_action=None):
            """
            Initialize a new state
            """
            self.state: CircuitStateDQN = state
            self.parent_state, self.parent_action = parent_state, parent_action
            self.r_previous = r_previous
            self.num_actions = len(self.state.device.edges)
            self.solution: np.ndarray = copy.copy(solution) if solution is not None else np.zeros(self.num_actions)

            self.rollout_reward = self.calc_rollout_reward() if self.parent_action is not None else 0.0
            self.action_mask = np.concatenate([self.state.device.swappable_edges(self.solution), np.array([False])])

            self.n_value = torch.zeros(self.num_actions + 1)
            self.q_value = torch.zeros(self.num_actions + 1)
            self.child_states: ty.List[ty.Optional[MCTSAgent.MCTSState]] = [None for _ in range(self.num_actions)]

            self.model = model
            self.priors, _ = self.model(self.state)

            self.priors += np.bitwise_not(self.action_mask) * -1e8
            self.priors = (torch.softmax(torch.tensor(self.priors), dim=0) *
                           torch.tensor(self.action_mask).float()).flatten()
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
            best_val = torch.max(uct)
            best_move_indices: torch.Tensor = torch.where(torch.eq(best_val, uct))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def prior_noise(self):
            """
            Adds dirichlet noise to priors.
            Called when the state is the root node
            """
            if sum(self.solution) < 1e-6:
                return False
            noise = np.random.dirichlet([self.HYPERPARAM_NOISE_ALPHA for _ in self.priors])
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise

        def calc_rollout_reward(self, num_rollouts=None):
            """
            performs R random rollout, the total reward in each rollout is computed.
            returns: mean across the R random rollouts.
            """
            if num_rollouts is None:
                next_state, _, _, _ = step(self.solution, self.state)
                _, self.rollout_reward = self.model(next_state)
                return self.rollout_reward
            else:
                total_reward = 0
                for i in range(num_rollouts):
                    solution = np.copy(self.solution)
                    while True:
                        mask = np.concatenate([self.state.device.swappable_edges(solution), np.array([False])])
                        if not np.any(mask):
                            break
                        swap = np.random.choice(np.where(mask)[0])
                        if swap == len(solution):
                            break
                        solution[swap] = True
                    _, reward, _, _ = step(self.solution, self.state)
                    total_reward += reward
                return total_reward / num_rollouts

    class MCTSStepper:
        """
        Monte Carlo Tree Search combiner object for evaluating the combination of moves
        that will form one step of the simulation.
        This at the moment does not look into the future steps, just calls an evaluator
        """

        HYPERPARAM_DISCOUNT_FACTOR = 0.95
        HYPERPARAM_EXPLORE_C = 0.01
        HYPERPARAM_POLICY_TEMPERATURE = 0

        def __init__(self, state, model):
            self.state = state
            self.model = model
            self.root = MCTSAgent.MCTSState(state, self.model)

        def search(self, n_mcts):
            """Perform the MCTS search from the root"""
            for _ in range(n_mcts):
                mcts_state = self.root  # reset to root for new trace

                while np.any(self.state.device.swappable_edges(mcts_state.solution)):
                    action_index: int = mcts_state.select()
                    if action_index == len(mcts_state.solution):
                        # MCTS Algorithm: SELECT STAGE - terminal action
                        mcts_state.update_q(mcts_state.r_previous + self.HYPERPARAM_DISCOUNT_FACTOR *
                                            mcts_state.rollout_reward, action_index)
                        break
                    elif mcts_state.child_states[action_index] is not None:
                        # MCTS Algorithm: SELECT STAGE
                        mcts_state = mcts_state.child_states[action_index]
                        continue
                    else:
                        # MCTS Algorithm: EXPAND STAGE
                        next_solution = np.copy(mcts_state.solution)
                        next_solution[action_index] = True
                        _next_state, reward, _done, _debug = step(next_solution, self.state)
                        mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                            self.state, self.model, next_solution, reward, (self, action_index))
                        mcts_state = mcts_state.child_states[action_index]
                        break

                # MCTS Algorithm: BACKUP STAGE
                total_reward = mcts_state.rollout_reward
                while mcts_state.parent_action is not None:
                    total_reward = mcts_state.r_previous + self.HYPERPARAM_DISCOUNT_FACTOR * total_reward
                    mcts_state = mcts_state.parent_state
                    mcts_state.update_q(total_reward, mcts_state.parent_action)

        def act(self):
            """Process the output at the root node"""
            while True:
                self.search(100)
                pos = torch.argmax(self.root.q_value).item()
                if self.root.child_states[pos] is None:
                    return self.root.solution
                else:
                    self.root = self.root.child_states[pos]

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def act(self, state: CircuitStateDQN):
        return self.MCTSStepper(state, self.model).act(), 0.0

    def replay(self, memory: ReplayMemory):
        pass
