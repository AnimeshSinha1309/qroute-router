import copy

import torch
import numpy as np

from qroute.environment.device import DeviceTopology
from qroute.environment.state import CircuitStateDQN
from qroute.algorithms.actanneal import AnnealerAct
from qroute.utils.histogram import histogram
import qroute.hyperparams


class ActorCriticAgent(torch.nn.Module):

    def __init__(self, device: DeviceTopology):
        """
        Initializes the graph network as a torch module and makes the
        architecture and the graph.
        :param device: the Topology to which the agent is mapping to
        """
        super(ActorCriticAgent, self).__init__()
        self.device: DeviceTopology = device  # For the action space
        self.actor_model = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) ** 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, len(self.device.edges)),
            torch.nn.Softmax(dim=-1)
        ).to(qroute.hyperparams.DEVICE)
        self.critic_model = torch.nn.Sequential(
            torch.nn.Linear(self.device.max_distance, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        ).to(qroute.hyperparams.DEVICE)
        self.current_optimizer = torch.optim.Adam(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()))
        self.annealer = AnnealerAct(self, device)

        self.gamma = 0.8
        self.epsilon = 0.01  # TODO: Fix this constant and figure out if needed

    def forward(self, current_state, next_state=None, solution=None):
        """
        Get the value function approximations for the given state representation
        :param current_state: the current state
        :param next_state: the next state as a result of the action
        :param solution: boolean vector representing the solution (swap mask)
        :return: int/float, the value function approximation
        """
        targets, dist = self.get_representation(current_state)

        probs = self.actor_model(targets.view(-1))
        value = self.critic_model(dist)

        if next_state is None:
            return probs.detach().numpy(), value.detach().item()
        else:
            solution = torch.from_numpy(solution).float()
            _, solution_dist = self.get_representation(next_state)
            # Assumes that the baseline is 70% of the Softmax can be covered at any time
            # TODO: Change the Softmax to get individual probs then normalize for softer values
            probs_value = torch.multiply(torch.dot(probs, solution), value) - value * 0.8
            critic_value = self.critic_model(solution_dist) - value
            return (probs_value + critic_value).detach().item()

    def act(self, state: CircuitStateDQN):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        :param state: the state of the environment
        :return: np.array of shape (len(device),), the chosen action mask after annealing
        """
        state = copy.copy(state)
        if np.random.rand() <= self.epsilon:
            action, value = self.generate_random_action(state), -1
        else:
            action, value = self.annealer.simulated_annealing(state)
        return action, -value

    def replay(self, memory, batch_size=32):
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
            probs, value = self(state)

            if done:
                target = torch.tensor(reward)
                policy_loss = torch.tensor(0)
                value_loss = torch.square(value - target)
            else:
                solution, value = self.annealer.simulated_annealing(next_state, search_limit=10)
                probs, solution = torch.from_numpy(probs), torch.from_numpy(solution)
                target = torch.tensor(reward + self.gamma * value)
                advantage = torch.square(torch.subtract(target, value))
                policy_loss = torch.sum(torch.multiply(probs, solution)).item() * advantage
                value_loss = torch.square(torch.subtract(target, value))

            absolute_errors.append(torch.abs(value - target).detach().item())

            self.current_optimizer.zero_grad()
            loss = (policy_loss + value_loss) * is_weight
            loss.requires_grad = True
            loss.backward()
            self.current_optimizer.step()

        memory.batch_update(tree_index, absolute_errors)

    def get_representation(self, state: CircuitStateDQN):
        """
        Obtains the state representation
        """

        nodes_to_target_nodes = state.target_nodes
        distance_vector = histogram(state.target_distance, self.device.max_distance, 1)
        distance_vector = torch.from_numpy(distance_vector).to(qroute.hyperparams.DEVICE).float()

        interaction_map = torch.zeros((len(self.device), len(self.device)))
        for idx, target in enumerate(nodes_to_target_nodes):
            if target == -1:
                continue
            interaction_map[idx, target] = 1

        return interaction_map, distance_vector
