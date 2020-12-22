"""
Annealer class for a Actor Critic model
"""

import copy
import numpy as np
import torch

from qroute.environment.device import DeviceTopology
from qroute.metas import CombinerAgent, TransformationState, ReplayMemory, MemoryItem


class AutoRegressor(CombinerAgent):
    """
    Class to perform simulated annealing using a policy gradient model + value function approximator
    """

    def __init__(self, agent, device):
        """
        Sets hyper-parameters and stores the agent and environment to initialize Annealer
        :param agent: Agent, to evaluate the value function
        :param device: environment, maintaining the device and state
        """
        self.device: DeviceTopology = device
        self.agent: torch.nn.Module = agent

    def act(self, state: TransformationState):
        state = copy.copy(state)
        solution, value = np.full(shape=len(self.device.edges), fill_value=False), 0
        while True:
            available = np.concatenate([self.device.swappable_edges(solution), np.array([True])])
            probs, value = self.agent(state)
            swap_edge = np.argmax(np.multiply(available, probs))
            if swap_edge == len(self.device.edges):  # If STOP action is selected.
                break
            solution[swap_edge] = True
        return solution, value

    def replay(self, memory: ReplayMemory):
        absolute_errors = []

        experience: MemoryItem
        for experience in memory:
            # Train the current model (model.fit in current state)
            probs, value = self.agent(experience.state)
            _, next_value = (None, 0) if experience.done else self.act(experience.next_state)

            probs = torch.from_numpy(probs[:-1])
            action = torch.from_numpy(experience.action)

            target = torch.tensor(experience.reward + self.agent.gamma * next_value)
            advantage = torch.subtract(target, value)
            policy_loss = torch.sum(probs * action).item() * advantage
            value_loss = torch.square(torch.subtract(target, value))

            absolute_errors.append(torch.abs(value - target).detach().item())

            self.agent.optimizer.zero_grad()
            loss = (policy_loss + value_loss)
            loss.requires_grad = True
            loss.backward()
            self.agent.optimizer.step()

        memory.clear()
