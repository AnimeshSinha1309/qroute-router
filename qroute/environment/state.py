import numpy as np

from qroute.environment.device import DeviceTopology
from qroute.environment.circuits import CircuitRepDQN
import qroute.hyperparams


class CircuitStateDQN:
    """
    Represents the State of the system when transforming a circuit. This holds the reference
    copy of the environment and the state of the transformation (even within a step).

    :param qubit_locations: The mapping array, tau
    :param qubit_targets: Next qubit location each qubit needs to interact with
    :param circuit_progress: Array keeping track of how many gates are executed by each qubit for updates
    :param protected_nodes: The nodes that are sealed because they are being operation in this step

    :param circuit: holds the static form of the circuit
    :param device: holds the device we are running the circuit on (for maintaining the mapping)
    """

    def __init__(self, circuit: CircuitRepDQN, device: DeviceTopology,
                 qubit_locations=None, qubit_targets=None, circuit_progress=None, protected_nodes=None):
        self.qubit_locations: np.ndarray = qubit_locations
        self.qubit_targets: np.ndarray = qubit_targets
        self.circuit_progress: np.ndarray = circuit_progress
        self.protected_nodes: set = protected_nodes
        # The state must have access to the overall environment
        self.circuit = circuit
        self.device = device

    def generate_starting_state(self):
        """
        Gets the state the DQN starts on. Randomly initializes the mapping if not specified
        otherwise, and sets the progress to 0 and gets the first gates to be scheduled.

        :return: list, [(n1, n2) next gates we can schedule]
        """
        # The starting state should be setup right
        self.qubit_locations = np.arange(len(self.circuit))
        print('Qubit Locations:', self.qubit_locations)
        np.random.shuffle(self.qubit_locations)

        self.protected_nodes = set()
        self.qubit_targets = np.array([targets[0] if len(targets) > 0 else -1 for targets in self.circuit])
        self.circuit_progress = np.zeros(len(self.circuit))

    # Running the circuit

    def next_requirements(self):
        """
        For each qubit, it assigns a next interaction if both qubits want to interact with each other.
        Assigns None in the interaction if the qubits do not interact with each other.

        :return gates: list of length n_qubits, (q1, q2) if both want to interact with each other, None otherwise
        """
        gates = [(q, self.qubit_targets[q]) if q == self.qubit_targets[self.qubit_targets[q]] and
                                               q < self.qubit_targets[q]
                 else None for q in range(0, len(self.qubit_targets))]
        return list(filter(lambda gate: gate is not None and gate[0] < gate[1], gates))

    def next_executables(self):
        """
        Takes the output of next gates, and returns only those which are executable on the hardware

        :return: list, [(q1, q2) where it's the next operation to perform and possible on hardware]
        """
        next_gates = self.next_requirements()
        return list(filter(self.device.is_adjacent, next_gates))

    def next_gates(self):
        """
        Gets the next set of gates we can execute as hardware nodes, and marks them as protected in the state.
        This function MUTATES the state.

        :return: list, [(n1, n2) next gates we can schedule]
        """
        # Converts the qubit-gates to the node gates
        next_gates_to_schedule = self.next_executables()
        next_gates_to_schedule_between_nodes = []
        for (q1, q2) in next_gates_to_schedule:
            (n1, n2) = (np.where(np.array(self.qubit_locations) == q1)[0][0],
                        np.where(np.array(self.qubit_locations) == q2)[0][0])
            gate_between_nodes = (n1, n2) if n1 < n2 else (n2, n1)
            next_gates_to_schedule_between_nodes.append(gate_between_nodes)
        # Makes those nodes as protected which are in the gate arrays
        protected_nodes = set()
        for (n1, n2) in next_gates_to_schedule_between_nodes:
            protected_nodes.add(n1)
            protected_nodes.add(n2)
        # Returns the gate array and updates protected nodes
        self.protected_nodes = protected_nodes
        return next_gates_to_schedule_between_nodes

    def execute_next(self):
        """
        Updates the state of the system with whatever interactions can be executed on the hardware.
        This function MUTATES the state.

        :return: int, reward gained by being able to schedule a gate
        """
        reward = 0
        for (q1, q2) in self.next_executables():
            # Increment the progress for both qubits by 1
            self.circuit_progress[q1] += 1
            self.circuit_progress[q2] += 1
            # Updates the qubit targets
            self.qubit_targets[q1] = self.circuit[q1][self.circuit_progress[q1]] \
                if self.circuit_progress[q1] < len(self.circuit[q1]) else -1
            self.qubit_targets[q2] = self.circuit[q2][self.circuit_progress[q2]] \
                if self.circuit_progress[q2] < len(self.circuit[q2]) else -1
            # The the reward for this gate which will be executed in next time step for sure, (q1, q2)
            reward += qroute.hyperparams.REWARD_GATE
        return reward

    def is_done(self):
        """
        Returns True iff each qubit has completed all of its interactions

        :return: bool, True if the entire circuit is executed
        """
        return all([target == -1 for target in self.qubit_targets])

    # State needs to help solution keep track of what it can do

    def swappable_edges(self, current_action):
        """
        List of edges that can be operated with swaps, given the current state and blocked edges

        :param current_action: list, boolean array of edges being currently swapped (current solution)
        :return: list, edges which can still be swapped
        """
        available_edges_mask = self.protected_edges

        # We block any edges connected to nodes already involved in a swap, except those actually being swapped
        current_action_nodes = set()
        for i, val in enumerate(current_action):
            if val == 1:
                (n1, n2) = self.device.edges[i]
                current_action_nodes.add(n1)
                current_action_nodes.add(n2)
        for idx, edge in enumerate(self.device.edges):
            block_1, block_2 = edge[0] in current_action_nodes, edge[1] in current_action_nodes
            if (block_1 or block_2) and not (block_1 and block_2):
                available_edges_mask[idx] = False

        return np.where(np.array(available_edges_mask) == 1)[0]

    # Other utility functions and properties

    def __copy__(self):
        """
        Makes a copy, keeping the reference to the same environment, but
        instantiating the rest of the state again.

        :return: State, a copy of the original, but independent of the first one, except env
        """
        return CircuitStateDQN(self.circuit, self.device, self.qubit_locations[:], self.qubit_targets[:],
                               self.circuit_progress[:], set(self.protected_nodes))

    def __eq__(self, other):
        """
        Checks whether two state are identical

        :param other: State, the other state to compare against
        :return: True if they are the same, False otherwise
        """
        return np.array_equal(self.qubit_locations, other.qubit_locations) and \
               np.array_equal(self.qubit_targets, other.qubit_targets) and \
               np.array_equal(self.circuit_progress, other.circuit_progress) and \
               self.protected_nodes == other.protected_nodes

    def calculate_distances_summary(self, qubit_locations, qubit_targets):
        """
        Get's all the distances for each qubits with next operation qubit

        :param qubit_locations: list/array, current mapping of logical to physical qubits
        :param qubit_targets: list/array, the next elements to match against
        :return: list, distances for each qubit on the next operation
        """
        distances = np.zeros(len(self.circuit))
        for q in range(len(self.circuit)):
            target_qubit = qubit_targets[q]
            if target_qubit == -1:
                distances[q] = np.inf
                continue
            node = np.where(np.array(qubit_locations) == q)[0][0]
            target_node = np.where(np.array(qubit_locations) == qubit_targets[q])[0][0]
            distances[q] = self.device.distances[node, target_node]
        return distances

    @property
    def protected_edges(self):
        """
        Make a list of edges which are blocked given nodes which are blocked
        :return: list, edges that are blocked
        """
        return np.array(list(map(
            lambda e: True if e[0] in self.protected_nodes or e[1] in self.protected_nodes else False,
            self.device.edges)))
