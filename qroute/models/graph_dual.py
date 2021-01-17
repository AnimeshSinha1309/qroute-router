import typing
import numpy as np

import torch
import torch_geometric

from qroute.environment.device import DeviceTopology
from qroute.environment.state import CircuitStateDQN


class GraphDualModel(torch.nn.Module):

    def __init__(self, device: DeviceTopology, stop_move: bool = False):
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
            torch.nn.Linear(len(self.device) * (4 + 1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 4, len(self.device.edges) + (1 if stop_move else 0)),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, state: CircuitStateDQN) -> typing.Tuple[np.ndarray, int]:
        """
        The callable for the model, does the forward propagation step
        :param state: input state of the circuit
        :return: the probability of each of the actions and value function for state
        """
        x, r = self.get_representation(state)
        x = self.edge_conv(x, self.edges)
        x = x.view(-1)
        r = torch.cat([x, r])
        policy = self.policy_head(x).detach().numpy()
        value: int = self.value_head(r).detach().item()
        # policy[-1] = -1e10  FIXME: Force this constraint for all other functions
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

        remaining_targets = torch.from_numpy(state.remaining_targets)

        return interaction_map, remaining_targets
