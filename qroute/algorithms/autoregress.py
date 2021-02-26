"""
Annealer class for a Actor Critic model
"""

import copy
import numpy as np
import torch

from ..environment.state import CircuitStateDQN
from ..metas import CombinerAgent, ReplayMemory


class AutoRegressor(CombinerAgent):
    """
    Class to perform simulated annealing using a policy gradient model + value function approximator
    """

    def __init__(self, model, device):
        """
        Sets hyper-parameters and stores the agent and environment to initialize Annealer
        :param model: Agent, to evaluate the value function
        :param device: environment, maintaining the device and state
        """
        super(AutoRegressor, self).__init__(model, device)

    def act(self, state: CircuitStateDQN):
        state = copy.copy(state)
        solution, value = np.full(shape=len(self.device.edges), fill_value=False), 0

        # Force at least 1 swap
        available = self.device.swappable_edges(solution)
        probs, value = self.model(state)
        swap_edge = np.argmax(np.multiply(available, probs[:-1]))
        solution[swap_edge] = True

        while True:
            available = np.concatenate([self.device.swappable_edges(solution), np.array([True])])
            probs, value = self.model(state)
            swap_edge = np.argmax(np.multiply(available, probs))
            if swap_edge == len(self.device.edges):  # If STOP action is selected.
                break
            solution[swap_edge] = True
        return solution, value

    def replay(self, memory: ReplayMemory):
        absolute_errors = []

        for experience in memory:
            # Train the current model (model.fit in current state)
            probs, value = self.model(experience.state)
            _, next_value = (None, 0) if experience.done else self.act(experience.next_state)

            probs = torch.from_numpy(probs[:-1])
            action = torch.from_numpy(experience.action)

            target = torch.tensor(experience.reward + self.model.gamma * next_value)
            advantage = torch.subtract(target, value)
            policy_loss = torch.sum(probs * action).item() * advantage
            value_loss = torch.square(torch.subtract(target, value))

            absolute_errors.append(torch.abs(value - target).detach().item())

            self.model.optimizer.zero_grad()
            loss = (policy_loss + value_loss)
            loss.requires_grad = True
            loss.backward()
            self.model.optimizer.step()

        memory.clear()
