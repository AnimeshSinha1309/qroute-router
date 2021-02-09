"""
Annealer class for a Actor Critic model
"""

import copy
import math
import numpy as np
import torch

from ..environment.state import CircuitStateDQN
from ..environment.env import step
from ..visualizers.solution_validator import check_valid_solution
from ..metas import CombinerAgent
from ..memory.per import MemoryPER


class AnnealerAct(CombinerAgent):
    """
    Class to perform simulated annealing using a policy gradient model + value function approximator
    """

    def __init__(self, model, device):
        """
        Sets hyper-parameters and stores the agent and environment to initialize Annealer
        :param model: Agent, to evaluate the value function
        :param device: environment, maintaining the device and state
        """
        super(AnnealerAct, self).__init__(model, device)
        self.initial_temperature = 60.0
        self.min_temperature = 0.1
        self.cooling_multiplier = 0.95

    def _generate_initial_solution(self, edge_probs):
        """
        Makes a random initial solution to start with by populating with whatever swaps possible
        :param edge_probs: vector with the probability values for each edge to swap
        :return: list, initial solution as boolean array of whether to swap each node
        """
        initial_solution = np.full(len(self.device.edges), fill_value=False)
        available_edges = self.device.swappable_edges(initial_solution)
        probs = np.multiply(available_edges, edge_probs)
        if not np.any(available_edges):
            return initial_solution
        edge_index_to_swap = np.random.choice(np.arange(len(self.device.edges)), p=probs/np.sum(probs))
        initial_solution[edge_index_to_swap] = (initial_solution[edge_index_to_swap] + 1) % 2
        return initial_solution

    def _get_neighbour_solution(self, current_solution):
        """
        Get a solution neighboring current, that is one swap inserted
        :param current_solution: list of edges to swap, current solution to start with
        :return: list, neighbor solution
        """
        neighbour_solution = copy.copy(current_solution)
        available_edges = np.bitwise_or(neighbour_solution, self.device.swappable_edges(neighbour_solution))
        if not np.any(available_edges):
            raise RuntimeError("Ran out of edges to swap")

        swap_edge = np.random.choice(np.arange(len(available_edges)), p=available_edges/np.sum(available_edges))
        neighbour_solution[swap_edge] = not neighbour_solution[swap_edge]
        check_valid_solution(neighbour_solution, self.device)
        return neighbour_solution

    @staticmethod
    def acceptance_probability(current_value, new_value, temperature):
        """
        Compute acceptance probability given delta-energy
        :param current_value: int/float, initial energy (negative of value function)
        :param new_value: int/float, final energy (negative of value function)
        :param temperature: int/float, temperature in the simulation (randomness)
        :return: int or float, probability to accept
        """
        if current_value < new_value:
            return 1
        else:
            energy_diff = current_value - new_value
            probability = math.exp(-energy_diff / temperature)
            return probability

    def act(self, state: CircuitStateDQN):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        :param state: the state of the environment
        :return: np.array of shape (len(device),), the chosen action mask after annealing
        """
        state = copy.copy(state)
        action, value = self.simulated_annealing(state)
        return action, -value

    def replay(self, memory: MemoryPER, batch_size=32):
        """
        Learns from past experiences
        :param memory: MemoryTree object, the experience buffer to sample from
        :param batch_size: number of experiences to sample from the experience buffer when training
        """
        tree_index, minibatch, is_weights = memory.sample(batch_size)
        absolute_errors = []
        is_weights = np.reshape(is_weights, -1)

        for experience, is_weight in zip(minibatch, is_weights):
            state, reward, next_state, done = experience
            # Train the current model (model.fit in current state)
            probs, value = self.model(state)

            if done:
                target = torch.tensor(reward)
                policy_loss = torch.tensor(0)
                value_loss = torch.square(value - target)
            else:
                solution, value = self.simulated_annealing(next_state, search_limit=10)
                probs, solution = torch.from_numpy(probs), torch.from_numpy(solution)
                target = torch.tensor(reward + self.model.gamma * value)
                advantage = torch.square(torch.subtract(target, value))
                policy_loss = torch.sum(torch.multiply(probs, solution)).item() * advantage
                value_loss = torch.square(torch.subtract(target, value))

            absolute_errors.append(torch.abs(value - target).detach().item())

            self.model.optimizer.zero_grad()
            loss = (policy_loss + value_loss) * is_weight
            loss.requires_grad = True
            loss.backward()
            self.model.optimizer.step()

        memory.batch_update(tree_index, absolute_errors)

    def simulated_annealing(self, current_state, search_limit=None):
        """
        Uses Simulated Annealing to find the next best state based on combinatorial
        actions taken by the agent.

        :param current_state: State, the state before this iterations of sim-anneal
        :param search_limit: int, max iterations to search for
        :return: best_solution, value of best_energy
        """
        temp_state: CircuitStateDQN = copy.copy(current_state)
        edge_probs, current_value = self.model(current_state)
        edge_probs = edge_probs  # Remove the STOP token element
        current_solution = self._generate_initial_solution(edge_probs)
        new_state: CircuitStateDQN = copy.copy(current_state)
        assert temp_state == new_state, "State not preserved when selecting action"

        if np.all(current_solution == 0):
            return current_solution, -np.inf

        temp = self.initial_temperature
        best_solution = copy.copy(current_solution)
        best_value = current_value

        iterations_since_best = 0
        iterations = 0

        while temp > self.min_temperature:
            if search_limit is not None and iterations > search_limit:
                break

            new_solution = self._get_neighbour_solution(current_solution)
            new_state, _, _, _ = step(new_solution, current_state)
            new_value = self.model.evaluate(current_state, new_state, new_solution)
            accept_prob = self.acceptance_probability(current_value, new_value, temp)

            if accept_prob > np.random.random():
                current_solution = new_solution
                current_value = new_value

                if current_value > best_value:
                    best_solution = copy.copy(current_solution)
                    best_value = current_value
                    iterations_since_best = 0

            temp = temp * self.cooling_multiplier
            iterations_since_best += 1
            iterations += 1

        return best_solution, best_value
