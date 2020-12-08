import torch
import numpy as np

from environment.device import DeviceTopology
from environment.state import CircuitStateDQN
from combiners.simanneal import AnnealerDQN


class DoubleDQNAgent(torch.nn.Module):

    def __init__(self, device: DeviceTopology):
        """
        Initializes the graph network as a torch module and makes the
        architecture and the graph.
        :param device: the Topology to which the agent is mapping to
        """
        super(DoubleDQNAgent, self).__init__()
        self.device: DeviceTopology = device  # For the action space
        self.current_model = torch.nn.Sequential(
            torch.nn.Linear(self.device.max_distance, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 1),
        )
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(self.device.max_distance, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 1),
        )
        self.current_optimizer = torch.optim.Adam(self.current_model.parameters())
        self.annealer = AnnealerDQN(self, device)

        self.gamma = 0.8
        self.epsilon_decay = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001

    def forward(self, dist_histogram):
        """
        Get the value function approximations for the given state representation

        :param dist_histogram: list, the histogram of distances in current state
        :return: int/float, the value function approximation
        """
        return self.current_model(dist_histogram)

    def act(self, current_state: CircuitStateDQN):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)

        :param current_state: the state of the environment
        :return: np.array of shape (len(device),), the chosen action mask after annealing
        """
        if np.random.rand() <= self.epsilon:
            action = self.generate_random_action(current_state.protected_nodes)
        else:
            # Choose an action using the agent's current neural network
            action, _ = self.annealer.simulated_annealing(current_state, action_chooser='model')
        return action

    def replay(self, batch_size):
        """
        Learns from past experiences

        :param batch_size: number of experiences to sample from the experience buffer when training
        """

        tree_index, minibatch, is_weights = self.memory_tree.sample(batch_size)
        absolute_errors = []

        for experience, is_weight in zip(minibatch, is_weights):
            [state, reward, next_state, done] = experience[0]
            target_nodes, next_target_nodes = self.obtain_target_nodes(state), self.obtain_target_nodes(next_state)
            q_val = self.current_model.predict(target_nodes)[0]
            target = reward + (self.gamma * self.target_model.predict(next_target_nodes)[0] if not done else 0)
            absolute_errors.append(abs(q_val - target))

            self.current_model.fit(target_nodes, [target], epochs=1, verbose=0, sample_weight=is_weight)

        self.memory_tree.batch_update(tree_index, absolute_errors)

        # Epsilon decay function - exploration vs. exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
