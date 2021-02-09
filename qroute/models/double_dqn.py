import random

import torch
import numpy as np

from ..environment.device import DeviceTopology
from ..environment.state import CircuitStateDQN
from ..algorithms.simanneal import AnnealerDQN
from ..hyperparams import DEVICE


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
            torch.nn.Linear(2 * self.device.max_distance, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        ).to(DEVICE)
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.device.max_distance, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        ).to(DEVICE)
        self.current_optimizer = torch.optim.Adam(self.current_model.parameters())
        self.annealer = AnnealerDQN(self, device)

        self.gamma = 0.8
        self.epsilon_decay = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001

    def update_target_model(self):
        """
        Copy weights from the current model to the target model
        """
        self.target_model.load_state_dict(self.current_model.state_dict())

    def forward(self, current_state, next_state, action_chooser='model'):
        """
        Get the value function approximations for the given state representation

        :param current_state: the current state
        :param next_state: the next state as a result of the action
        :param action_chooser: str, model if current model or target if target model
        :return: int/float, the value function approximation
        """
        current_distance_vector = self.get_distance_metric(current_state)
        next_distance_vector = self.get_distance_metric(next_state)

        nn_input = torch.cat([current_distance_vector, next_distance_vector], dim=-1)

        if action_chooser == 'model':
            q_val = self.current_model(nn_input)
        elif action_chooser == 'target':
            q_val = self.target_model(nn_input)
        else:
            raise ValueError('Action_chooser must be either model or target')

        return q_val

    def act(self, state: CircuitStateDQN):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)

        :param state: the state of the environment
        :return: np.array of shape (len(device),), the chosen action mask after annealing
        """
        if np.random.rand() <= self.epsilon:
            action, value = self.generate_random_action(state), -1
        else:
            action, value = self.annealer.simulated_annealing(state, action_chooser='model')
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
            [state, reward, next_state, done] = experience[0]
            # Train the current model (model.fit in current state)
            q_val = self(state, next_state)[0]

            if done:
                target = reward
            else:
                _, energy = self.annealer.simulated_annealing(next_state, action_chooser='target', search_limit=10)
                target = reward - self.gamma * energy

            absolute_errors.append(abs(q_val.detach() - target))

            self.current_optimizer.zero_grad()
            loss = torch.multiply(torch.square(torch.subtract(q_val, target)), is_weight)
            loss.backward()
            self.current_optimizer.step()

        memory.batch_update(tree_index, absolute_errors)

        # Epsilon decay function - exploration vs. exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def generate_random_action(self, state: CircuitStateDQN):
        """
        Generates a random layer of swaps. Care is taken to ensure that all swaps can occur in parallel.
        That is, no two neighbouring edges undergo a swap simultaneously.
        """
        action = np.array([0] * len(self.device.edges))  # an action representing an empty layer of swaps

        edges = [(n1, n2) for (n1, n2) in self.device.edges]
        edges = list(filter(lambda e: e[0] not in state.protected_nodes and e[1] not in state.protected_nodes, edges))
        edge_index_map = {edge: index for index, edge in enumerate(edges)}

        while len(edges) > 0:
            edge, action[edge_index_map[edge]] = random.sample(edges, 1)[0], 1
            edges = [e for e in edges if e[0] not in edge and e[1] not in edge]
        return action

    def get_distance_metric(self, state: CircuitStateDQN):
        """
        Obtains a vector that summarises the different distances from qubits to their targets.
        More precisely, x_i represents the number of qubits that are currently a distance of i away from their targets.
        If there are n qubits, then the length of this vector will also be n.
        """
        nodes_to_target_qubits = [
            state._qubit_targets[state.node_to_qubit[n]] for n in range(0, len(state.node_to_qubit))]
        nodes_to_target_nodes = [
            next(iter(np.where(np.array(state.node_to_qubit) == q)[0]), -1) for q in nodes_to_target_qubits]

        distance_vector = np.zeros(self.device.max_distance)

        for node in range(len(nodes_to_target_nodes)):
            target = nodes_to_target_nodes[node]
            if target == -1:
                continue
            d = int(self.device.distances[node, target])
            distance_vector[d - 1] += 1

        distance_vector = torch.from_numpy(distance_vector).to(DEVICE).float()
        return distance_vector
