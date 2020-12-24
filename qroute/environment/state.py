import numpy as np

from qroute.environment.device import DeviceTopology
from qroute.environment.circuits import CircuitRepDQN
from qroute.metas import TransformationState


class CircuitStateDQN(TransformationState):
    """
    Represents the State of the system when transforming a circuit. This holds the reference
    copy of the environment and the state of the transformation (even within a step).

    :param node_to_qubit: The mapping array, tau
    :param qubit_targets: Next qubit location each qubit needs to interact with
    :param circuit_progress: Array keeping track of how many gates are executed by each qubit for updates
    :param circuit: holds the static form of the circuit
    :param device: holds the device we are running the circuit on (for maintaining the mapping)
    """

    def __init__(self, circuit: CircuitRepDQN, device: DeviceTopology, node_to_qubit=None,
                 qubit_targets=None, circuit_progress=None):
        """
        Gets the state the DQN starts on. Randomly initializes the mapping if not specified
        otherwise, and sets the progress to 0 and gets the first gates to be scheduled.
        :return: list, [(n1, n2) next gates we can schedule]
        """
        # The state must have access to the overall environment
        self.circuit = circuit
        self.device = device
        assert len(circuit) == len(device), "All qubits on target device or not used, or too many are used"
        # The starting state should be setup right
        self._node_to_qubit = np.random.permutation(len(self.circuit)) \
            if node_to_qubit is None else node_to_qubit
        self._qubit_targets = np.array([targets[0] if len(targets) > 0 else -1 for targets in self.circuit.circuit]) \
            if qubit_targets is None else qubit_targets
        self._circuit_progress = np.zeros(len(self.circuit), dtype=np.int) \
            if circuit_progress is None else circuit_progress

    def execute_swap(self, solution):
        """
        Updates the state of the system with whatever swaps are executed in the solution.
        This function MUTATES the state.
        :param solution: boolean np.array, whether to take each edge on the device
        :return list of pairs, pairs of nodes representing gates which will be executed
        """
        gates_being_executed = []
        for edge, sol in zip(self.device.edges, solution):
            if sol:
                node1, node2 = edge
                gates_being_executed.append(edge)
                self._node_to_qubit[node1], self._node_to_qubit[node2] = \
                    self._node_to_qubit[node2], self._node_to_qubit[node1]
        return gates_being_executed

    def execute_cnot(self):
        """
        Updates the state of the system with whatever interactions can be executed on the hardware.
        This function MUTATES the state.
        :return list of pairs, pairs of nodes representing gates which will be executed
        """
        gates_being_executed = []
        for (n1, n2) in self.device.edges:
            # Check if we want to execute CNOT on this edge
            q1, q2 = self._node_to_qubit[n1], self._node_to_qubit[n2]
            if self._qubit_targets[q1] != q2 or self._qubit_targets[q2] != q1:
                continue
            gates_being_executed.append((n1, n2))
            # Increment the progress for both qubits by 1
            self._circuit_progress[q1] += 1
            self._circuit_progress[q2] += 1
            # Updates the qubit targets
            self._qubit_targets[q1] = self.circuit[q1][self._circuit_progress[q1]] \
                if self._circuit_progress[q1] < len(self.circuit[q1]) else -1
            self._qubit_targets[q2] = self.circuit[q2][self._circuit_progress[q2]] \
                if self._circuit_progress[q2] < len(self.circuit[q2]) else -1
        return gates_being_executed

    def is_done(self):
        """
        Returns True iff each qubit has completed all of its interactions
        :return: bool, True if the entire circuit is executed
        """
        return np.all(self._qubit_targets == -1)

    # Other utility functions and properties

    def __copy__(self):
        """
        Makes a copy, keeping the reference to the same environment, but
        instantiating the rest of the state again.

        :return: State, a copy of the original, but independent of the first one, except env
        """
        return CircuitStateDQN(self.circuit, self.device, np.copy(self._node_to_qubit),
                               np.copy(self._qubit_targets), np.copy(self._circuit_progress))

    # noinspection PyProtectedMember
    def __eq__(self, other):
        """
        Checks whether two state are identical

        :param other: State, the other state to compare against
        :return: True if they are the same, False otherwise
        """
        return np.array_equal(self._node_to_qubit, other._node_to_qubit) and \
               np.array_equal(self._qubit_targets, other._qubit_targets) and \
               np.array_equal(self._circuit_progress, other._circuit_progress)

    @property
    def target_nodes(self):
        qubit_to_node = np.zeros(len(self._node_to_qubit), dtype=np.int)
        for i, v in enumerate(self._node_to_qubit):
            qubit_to_node[v] = i
        target_nodes = np.zeros(len(self._node_to_qubit), dtype=np.int)
        for i, v in enumerate(self._qubit_targets):
            target_nodes[qubit_to_node[i]] = qubit_to_node[v]
        return target_nodes

    @property
    def target_distance(self):
        qubit_to_node = np.zeros(len(self._node_to_qubit), dtype=np.int)
        for i, v in enumerate(self._node_to_qubit):
            qubit_to_node[v] = i
        target_distances = np.zeros(len(self._node_to_qubit), dtype=np.int)
        for i, v in enumerate(self._qubit_targets):
            target_distances[i] = self.device.distances[qubit_to_node[i], qubit_to_node[v]]
        return target_distances

    @property
    def node_to_qubit(self):
        return np.copy(self._node_to_qubit)
