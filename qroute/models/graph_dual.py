import typing
import numpy as np

import torch
import torch_geometric

from qroute.environment.device import DeviceTopology
from qroute.environment.state import CircuitStateDQN


class GraphDualModel(torch.nn.Module):

    def __init__(self, device: DeviceTopology):
        """
        Create the decision model for the given device topology
        :param device: the device object on which the agent should propose actions
        """
        super(GraphDualModel, self).__init__()
        self.device = device
        mlp = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.Softmax(dim=-1),
        )
        self.edge_conv = torch_geometric.nn.EdgeConv(aggr='add', nn=mlp)
        self.edges = torch.tensor(self.device.edges).transpose(1, 0)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 4, len(self.device.edges)),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, state: CircuitStateDQN) -> typing.Tuple[np.ndarray, int]:
        """
        The callable for the model, does the forward propagation step
        :param state: input state of the circuit
        :return: the probability of each of the actions and value function for state
        """
        x = self.get_representation(state)
        x = self.edge_conv(x, self.edges)
        x = x.view(-1)
        policy: np.ndarray = self.policy_head(x).detach().numpy()
        value: int = self.value_head(x).detach().item()
        return policy, value

    def get_representation(self, state: CircuitStateDQN):
        """
        Obtains the state representation
        :param state: the state of the circuit right now
        """
        nodes_to_target_nodes = state.target_nodes
        interaction_map = torch.zeros((len(self.device), len(self.device)))
        for idx, target in enumerate(nodes_to_target_nodes):
            if target == -1:
                continue
            interaction_map[idx, target] = 1
        return interaction_map
