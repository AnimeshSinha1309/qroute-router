"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing as ty

import numpy as np
import torch

from ..metas import CombinerAgent, ReplayMemory
from ..environment.state import CircuitStateDQN
from ..environment.env import step, evaluate


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
            self.model = model
            self.parent_state, self.parent_action = parent_state, parent_action
            self.r_previous = r_previous
            self.num_actions = len(self.state.device.edges)
            self.solution: np.ndarray = copy.copy(solution) if solution is not None else \
                np.full(self.num_actions, False)

            assert not np.any(np.bitwise_and(self.state.locked_edges, self.solution)), "Bad Action"

            self.rollout_reward = self.calc_rollout_reward() if self.parent_action is not None else 0.0
            self.action_mask = np.concatenate([state.device.swappable_edges(self.solution, self.state.locked_edges),
                                               np.array([solution is not None])])

            self.n_value = torch.zeros(self.num_actions + 1)
            self.q_value = torch.zeros(self.num_actions + 1)
            self.child_states: ty.List[ty.Optional[MCTSAgent.MCTSState]] = [None for _ in range(self.num_actions + 1)]

            self.priors, _ = self.model(self.state)

            self.priors += np.bitwise_not(self.action_mask) * -1e8
            self.priors = torch.flatten(torch.tensor(self.priors))  # TODO: is softmax needed?
            self.prior_noise()

        def update_q(self, reward, index):
            """
            Updates the q-value for the state
            :param reward: The obtained total reward from this state
            :param index: the index of the action chosen for which the reward was provided

            n_value is the number of times a node visited
            q_value is the q function

            n += 1, w += reward, q = w / n -> this is being implicitly computed using the weighted average
            """
            self.q_value[index] = (self.q_value[index] * self.n_value[index] + reward) / (self.n_value[index] + 1)
            self.n_value[index] += 1

        def select(self, c=1000) -> int:
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
            noise = np.random.dirichlet([self.HYPERPARAM_NOISE_ALPHA for _ in self.priors]) * self.action_mask
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise

        def calc_rollout_reward(self, num_rollouts=None):  # TODO: Benchmark this on 100 rollouts
            """
            performs R random rollout, the total reward in each rollout is computed.
            returns: mean across the R random rollouts.
            """
            if num_rollouts is None:
                assert not np.any(np.bitwise_and(self.state.locked_edges, self.solution)), "Bad Action"
                next_state, _, _, _ = step(self.solution, self.state)
                _, self.rollout_reward = self.model(next_state)
                return self.rollout_reward
            else:
                total_reward = 0
                for i in range(num_rollouts):
                    solution = np.copy(self.solution)
                    while True:
                        mask = np.concatenate([self.state.device.swappable_edges(solution, self.state.locked_edges),
                                               np.array([True])])
                        if not np.any(mask):
                            break
                        swap = np.random.choice(np.where(mask)[0])
                        if swap == len(solution):
                            break  # This only evaluates one step deep
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

        HYPERPARAM_DISCOUNT_FACTOR = 1.00  # No discounting for single stage MCTS
        HYPERPARAM_EXPLORE_C = 100
        HYPERPARAM_POLICY_TEMPERATURE = 0

        def __init__(self, state, model):
            self.state = state
            self.model = model
            self.root = MCTSAgent.MCTSState(state, self.model)

        def search(self, n_mcts):
            """Perform the MCTS search from the root"""
            max_depth, mean_depth = 0, 0

            for _ in range(n_mcts):
                mcts_state: MCTSAgent.MCTSState = self.root  # reset to root for new trace
                # input(str(self.root.n_value) + " " + str(self.root.q_value))  # To Debug the tree
                depth = 0

                while True:
                    depth += 1

                    action_index: int = mcts_state.select()
                    if action_index != len(mcts_state.solution):
                        assert not mcts_state.state.locked_edges[action_index], "Selecting a Bad Action"

                    if mcts_state.child_states[action_index] is not None:
                        # MCTS Algorithm: SELECT STAGE
                        mcts_state = mcts_state.child_states[action_index]
                        continue
                    else:
                        # MCTS Algorithm: EXPAND STAGE
                        if action_index == len(mcts_state.solution):
                            next_state, _reward, _done, _debug = step(mcts_state.solution, mcts_state.state)
                            mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                                next_state, self.model,
                                r_previous=0, parent_state=mcts_state, parent_action=action_index)
                        else:
                            next_solution = np.copy(mcts_state.solution)
                            next_solution[action_index] = True
                            reward = evaluate(next_solution, mcts_state.state) - \
                                     evaluate(mcts_state.solution, mcts_state.state)
                            mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                                mcts_state.state, self.model, next_solution, reward, mcts_state, action_index)
                        mcts_state = mcts_state.child_states[action_index]
                        break

                # MCTS Algorithm: BACKUP STAGE
                total_reward = mcts_state.rollout_reward
                while mcts_state.parent_action is not None:
                    total_reward = mcts_state.r_previous + self.HYPERPARAM_DISCOUNT_FACTOR * total_reward
                    mcts_state.parent_state.update_q(total_reward, mcts_state.parent_action)
                    mcts_state = mcts_state.parent_state

                max_depth = max(max_depth, depth)
                mean_depth += depth / n_mcts

            return max_depth, mean_depth

        def act(self):
            """Process the output at the root node"""
            while True:
                self.search(100)
                pos = self.root.select()
                if pos == len(self.root.solution) or self.root.child_states[pos] is None:
                    assert not np.any(np.bitwise_and(self.state.locked_edges, self.root.solution)), "Bad Action"
                    return self.root.solution
                else:
                    self.root = self.root.child_states[pos]

    def __init__(self, model, device):
        super(MCTSAgent, self).__init__(model, device)

    def act(self, state: CircuitStateDQN):
        solution = self.MCTSStepper(state, self.model).act()
        return solution, 0.0

    def replay(self, memory: ReplayMemory):
        pass
