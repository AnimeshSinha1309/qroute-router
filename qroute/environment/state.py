import qroute


class State:
    """
    The state is maintained as a graph of circuit and the mapping of logical to physical qubits

    Notes
    -----
    * The term "leaf" is used to refer to the layer of operations LC_0 which will be executed next.
      Leaf neighbors however, may encompass more than LC_0 if the node being queried has no operations in the
      leaf layer, then the leaf neighbor is the first layer where it's used in an operation.
    """

    def __init__(self, device, circuit):
        self.device: qroute.environment.device.IBMqx5Device = device
        self.circuit = circuit
        # Reward functions parameters
        self.gate_reward = 20
        self.distance_reduction_reward = 2
        self.negative_reward = -10
        self.circuit_completion_reward = 100
        self.alternative_reward_delivery = False

    def __copy__(self):
        pass

    @property
    def circuit_dag(self):
        return None

    @property
    def leaf_neighbors(self):
        return None

    @property
    def leaf_ops(self):
        return None
