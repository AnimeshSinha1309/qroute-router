import typing
import numpy as np

import torch
import torch_geometric

from ..environment.device import DeviceTopology
from ..environment.state import CircuitStateDQN


class NormActivation(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        tensor = tensor ** 2
        length = tensor.sum(dim=self.dim, keepdim=True)
        return tensor / length


class GraphDualModel(torch.nn.Module):

    def __init__(self, device: DeviceTopology, stop_move: bool = False):
        """
        Create the decision model for the given device topology
        :param device: the device object on which the agent should propose actions
        """
        super(GraphDualModel, self).__init__()
        self.device = device
        mlp = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 2, 50),
            torch.nn.SiLU(),
            torch.nn.Linear(50, 10),
            torch.nn.SiLU(),
            torch.nn.Linear(10, 4),
            torch.nn.SiLU(),
        )
        self.edge_conv = torch_geometric.nn.EdgeConv(aggr='add', nn=mlp)
        self.edges = torch.tensor(self.device.edges).transpose(1, 0)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 4 + len(self.device) + len(self.device.edges), 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(len(self.device) * 4 + len(self.device.edges),
                            len(self.device.edges) + (1 if stop_move else 0)),
            NormActivation(dim=-1),
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, state: CircuitStateDQN) -> typing.Tuple[int, np.ndarray]:
        """
        The callable for the model, does the forward propagation step
        :param state: input state of the circuit
        :return: the probability of each of the actions and value function for state
        """
        x, remaining, locks = self.get_representation(state)
        x = self.edge_conv(x, self.edges)
        x = x.view(-1)
        value_input = torch.cat([x, remaining, locks])
        policy_input = torch.cat([x, locks])
        policy = self.policy_head(policy_input)
        value: int = self.value_head(value_input)
        # policy[-1] = -1e10  FIXME: Force this constraint for all other functions
        return value, policy

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
        mutex_locks = torch.from_numpy(state.locked_edges)
        return interaction_map, remaining_targets, mutex_locks

    @staticmethod
    def _loss_p(predicted, target):
        loss = torch.sum(-target * ((1e-8 + predicted).log()))
        return loss

    @staticmethod
    def _loss_v(predicted, target):
        criterion = torch.nn.MSELoss()
        loss = criterion(predicted, target)
        return loss

    def fit(self, state, v, p):
        self.optimizer.zero_grad()
        self.train()
        v = v.reshape(1)
        pred_v, pred_p = self(state)
        v_loss = self._loss_v(pred_v, v)
        p_loss = self._loss_p(pred_p, p)
        loss = v_loss + p_loss
        loss.backward()
        self.optimizer.step()
        return v_loss.item(), p_loss.item()

