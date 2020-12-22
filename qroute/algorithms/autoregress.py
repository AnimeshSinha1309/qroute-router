"""
Annealer class for a Actor Critic model
"""

import copy
import numpy as np
import torch

from qroute.environment.device import DeviceTopology
from qroute.metas import CombinerAgent, TransformationState


class AnnealerAct(CombinerAgent):
    """
    Class to perform simulated annealing using a policy gradient model + value function approximator
    """

    def __init__(self, agent, device, memory):
        """
        Sets hyper-parameters and stores the agent and environment to initialize Annealer
        :param agent: Agent, to evaluate the value function
        :param device: environment, maintaining the device and state
        """
        self.device: DeviceTopology = device
        self.agent: torch.nn.Module = agent
        self.memory = memory

    def act(self, state: TransformationState):
        state = copy.copy(state)
        solution = np.full(shape=len(self.device.edges), fill_value=False)
        while True:
            available = np.concatenate([self.device.swappable_edges(solution), np.array([True])])
            result = self.agent(state)
            swap_edge = np.argmax(np.multiply(available, result))
            if swap_edge == self.device.edges:
                break
            solution[swap_edge] = True
        return solution

    def replay(self):
        minibatch = self.memory.sample()
        absolute_errors = []

        for experience in minibatch:
            state, reward, action, next_state, done = experience
            # Train the current model (model.fit in current state)
            probs, value = self.agent(state)
            _, next_value = (None, 0) if done else self.act(state)
            target = torch.tensor(reward + self.agent.gamma * value)
            advantage = torch.square(torch.subtract(target, value))
            policy_loss = torch.sum(torch.multiply(probs, solution)).item() * advantage
            value_loss = torch.square(torch.subtract(target, value))

            absolute_errors.append(torch.abs(value - target).detach().item())

            self.agent.optimizer.zero_grad()
            loss = (policy_loss + value_loss) * is_weight
            loss.requires_grad = True
            loss.backward()
            self.agent.optimizer.step()

        memory.batch_update(tree_index, absolute_errors)

