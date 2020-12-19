import numpy as np

from qroute.environment.device import DeviceTopology
from qroute.environment.circuits import CircuitRepDQN


class CircuitStateDQN:
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

    def next_requirements(self):
        """
        For each qubit, it assigns a next interaction if both qubits want to interact with each other.
        Assigns None in the interaction if the qubits do not interact with each other.
        :return gates: list of length n_qubits, (q1, q2) if both want to interact with each other, None otherwise
        """
        gates = [(q, self._qubit_targets[q]) if q == self._qubit_targets[self._qubit_targets[q]] and
                                                q < self._qubit_targets[q]
                 else None for q in range(0, len(self._qubit_targets))]

        return list(filter(lambda gate: gate is not None and gate[0] < gate[1], gates))

    def next_executables(self):
        """
        Takes the output of next gates, and returns only those which are executable on the hardware
        :return: list, [(q1, q2) where it's the next operation to perform and possible on hardware]
        """
        next_gates = self.next_requirements()

        def check_qubit_adjacency(qubits):
            q1, q2 = qubits
            node1 = np.where(np.array(self._node_to_qubit) == q1)[0][0]
            node2 = np.where(np.array(self._node_to_qubit) == q2)[0][0]
            return self.device.is_adjacent((node1, node2))

        return list(filter(check_qubit_adjacency, next_gates))

    def next_gates(self):
        """
        Gets the next set of gates we can execute as hardware nodes, and marks them as protected in the state.
        This function MUTATES the state.
        :return: list, [(n1, n2) next gates we can schedule]
        """
        # Converts the qubit-gates to the node gates
        qubit_gates = self.next_executables()
        node_gates = []
        for (q1, q2) in qubit_gates:
            (n1, n2) = (np.where(np.array(self._node_to_qubit) == q1)[0][0],
                        np.where(np.array(self._node_to_qubit) == q2)[0][0])
            gate_between_nodes = (n1, n2) if n1 < n2 else (n2, n1)
            node_gates.append(gate_between_nodes)
        return node_gates

    def execute_swap(self, solution):
        """
        Updates the state of the system with whatever swaps are executed in the solution.
        This function MUTATES the state.
        :param solution: boolean np.array, whether to take each edge on the device
        """
        for edge, sol in zip(self.device.edges, solution):
            node1, node2 = edge
            self._node_to_qubit[node1], self._node_to_qubit[node2] = \
                self._node_to_qubit[node2], self._node_to_qubit[node1]

    def execute_next(self):
        """
        Updates the state of the system with whatever interactions can be executed on the hardware.
        This function MUTATES the state.
        """
        gates_to_execute = self.next_executables()
        for (q1, q2) in gates_to_execute:
            # Increment the progress for both qubits by 1
            self._circuit_progress[q1] += 1
            self._circuit_progress[q2] += 1
            # Updates the qubit targets
            self._qubit_targets[q1] = self.circuit[q1][self._circuit_progress[q1]] \
                if self._circuit_progress[q1] < len(self.circuit[q1]) else -1
            self._qubit_targets[q2] = self.circuit[q2][self._circuit_progress[q2]] \
                if self._circuit_progress[q2] < len(self.circuit[q2]) else -1
        return gates_to_execute

    def is_done(self):
        """
        Returns True iff each qubit has completed all of its interactions
        :return: bool, True if the entire circuit is executed
        """
        return all([target == -1 for target in self._qubit_targets])

    # State needs to help solution keep track of what it can do

    def swappable_edges(self, current_action):
        """
        List of edges that can be operated with swaps, given the current state and blocked edges
        :param current_action: list, boolean array of edges being currently swapped (current solution)
        :return: list, edges which can still be swapped
        """
        available_edges_mask = np.full(shape=len(self.device.edges), fill_value=True)

        # We block any edges connected to nodes already involved in a swap, except those actually being swapped
        current_action_nodes = set()
        for i, used in enumerate(current_action):
            if used:
                (n1, n2) = self.device.edges[i]
                current_action_nodes.add(n1)
                current_action_nodes.add(n2)
        for idx, edge in enumerate(self.device.edges):
            if edge[0] in current_action_nodes or edge[1] in current_action_nodes:
                available_edges_mask[idx] = False
        for idx, used in enumerate(current_action):
            if used:
                available_edges_mask[idx] = True

        return available_edges_mask

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
