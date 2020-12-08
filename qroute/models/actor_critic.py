import torch
import numpy as np

from environment.device import DeviceTopology


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
            torch.nn.Linear(self.device.max_distance, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, len(self.device)),
        )
        self.critic_model = torch.nn.Sequential(
            torch.nn.Linear(self.device.max_distance, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 1),
        )

    def forward(self, dist_histogram):
        self.actor_model(dist_histogram)

    def act(self, current_state):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        """

        protected_nodes = current_state[3]

        if np.random.rand() <= self.epsilon:
            action = self.generate_random_action(protected_nodes)
            return action, "Random"

        # Choose an action using the agent's current neural network
        action, _ = self.annealer.simulated_annealing(current_state, action_chooser='model')
        return action, "Model"

    def replay(self, batch_size):
        """
        Learns from past experiences
        """

        tree_index, minibatch, is_weights = self.memory_tree.sample(batch_size)
        minibatch_with_weights = zip(minibatch, is_weights)
        absolute_errors = []

        for experience, is_weight in minibatch_with_weights:
            [state, reward, next_state, done] = experience[0]

            target_nodes = self.obtain_target_nodes(state)
            next_target_nodes = self.obtain_target_nodes(next_state)

            q_val = self.current_model.predict(target_nodes)[0]

            if done:
                target = reward
            else:
                target = reward + self.gamma * self.target_model.predict(next_target_nodes)[0]

            absolute_error = abs(q_val - target)
            absolute_errors.append(absolute_error)

            self.current_model.fit(target_nodes, [target], epochs=1, verbose=0, sample_weight=is_weight)

        self.memory_tree.batch_update(tree_index, absolute_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

