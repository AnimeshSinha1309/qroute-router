import torch
from environment.state import CircuitStateDQN


class DoubleDQNAgent:

    def __init__(self, circuit: CircuitStateDQN):
        """
        Initializes the graph network as a torch module and makes the
        architecture and the graph.
        :param circuit:
        """
        super(DoubleDQNAgent, self).__init__()

