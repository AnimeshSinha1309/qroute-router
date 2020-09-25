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

    def __init__(self, device, circuit, mapping=None):
        self.device: qroute.environment.device.DeviceTopology = device
        self.circuit: qroute.environment.circuits.CircuitState = circuit
        self.mapping: list = [i if i < len(circuit) else None for i in range(len(device))] if not mapping else mapping
        self.reward = None
        self.__operands_to_gate = {val: idx for idx, val in enumerate(self.circuit.gate_operands)}

    def execute_executable(self):
        """
        Execute any operations which can be popped from the leaf of the circuit without violating the
        device constraints
        :return: set, a list of all operations executed in this run
        """
        leaf_nodes = self.circuit.leaf_operations
        executed = set()
        for operation_id in leaf_nodes:
            operation = self.circuit.get_gate_operands(operation_id)
            if len(operation) == 1:
                self.circuit.pop(self.__operands_to_gate[operation])
                executed.add(operation)
            elif len(operation) == 2:
                q1, q2 = operation
                if self.device.distances[self.mapping.index(q1)][self.mapping.index(q2)] == 1:
                    self.circuit.pop(self.__operands_to_gate[operation])
                    executed.add(operation)
            else:
                raise ValueError('Multi-Qubit (> 2) gates are not operation primitives')
        return executed

    def execute_swaps(self, swaps):
        """
        Attempt to execute qubit swaps on the device
        :param swaps: list of 2-tuples, the hardware positions of the qubits that need to be swapped
        :return: None
        :raises: ValueError, if the swap operation is not permitted by the device
        """
        for swap in swaps:
            if self.device.swap_dist[swap[0]][swap[1]] <= 1:
                self.mapping[swap[0]], self.mapping[swap[1]] = self.mapping[swap[1]], self.mapping[swap[0]]
            else:
                raise ValueError('Qubits being swapped should currently be neighbors on the hardware')

    def execute_all(self, swaps):
        """
        Pops all gates which can be executed immediately and Executes Swap operations
        :param swaps: list of 2-tuples, the hardware positions of the qubits that need to be swapped
        :return: set, a list of all operations executed in this run
        :raises: ValueError, if the swap operation is not permitted by the device
        """
        self.execute_swaps(swaps)
        return self.execute_executable()
