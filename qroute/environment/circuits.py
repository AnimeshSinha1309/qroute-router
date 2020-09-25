import cirq
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class CircuitState:
    """
    Maintains the state of the circuit, allows evolving it as operations at the leaf
    keep getting executed
    """

    def __init__(self, circuit: cirq.Circuit):
        """
        Takes in a circuit (logical circuit - LC) and initializes the state object
        which will maintain the leaf nodes (operations which can be executed without dependencies)
        and the next node that each element wants to qubit with (for swapping heuristics)
        :param circuit: cirq.Circuit, The input logical circuit
        """
        self.circuit = circuit
        # TODO: Make this DAG object and other stuff global and shared between all instances, immutable
        self.dag = cirq.CircuitDag.from_circuit(self.circuit)
        self.dag_nodes = list(nx.topological_sort(self.dag))
        self.__node_to_index = {node: idx for idx, node in enumerate(self.dag_nodes)}
        # Make the queue of leaf nodes
        self.__leaf_queue: set = set()
        self.__indegree = np.zeros(shape=self.dag.number_of_nodes(), dtype=np.int16)
        for u in self.dag_nodes:
            for v in self.dag.successors(u):
                self.__indegree[self.__node_to_index[v]] += 1
        for idx, indegree in enumerate(self.__indegree):
            if self.__indegree[idx] == 0:
                self.__leaf_queue.add(idx)
        # Keep the map of next operation for each qubit
        qubits = self.circuit.all_qubits()
        self.__qubit_to_index = {qubit: idx for idx, qubit in enumerate(sorted(qubits))}
        self.__qubit_operations: list = [[] for _ in range(len(qubits))]
        self.__qubit_progress = np.zeros(shape=len(qubits), dtype=np.int16)
        operations = self.gate_operands
        for idx, op in enumerate(operations):
            for bit in op:
                neighbors = self.get_gate_operands(idx)
                if len(neighbors) == 1:
                    self.__qubit_operations[bit].append(bit)
                elif len(neighbors) == 2:
                    self.__qubit_operations[bit].append(neighbors[0] + neighbors[1] - bit)
                else:
                    raise ValueError('Cannot handle 3-Operand primitive gates')
        for idx, val in enumerate(self.__qubit_operations):
            if len(val) == 0:
                self.__qubit_progress[idx] = -1

    def __len__(self):
        return len(self.circuit.all_qubits())

    def draw_circuit_graph(self):
        """
        Draws the circuit as graph (attempts planar layout)
        :return: None
        """
        nx.draw(self.dag, pos=nx.planar_layout(self.dag), with_labels=False)
        plt.show()

    def pop(self, n):
        """
        Removes the n-th operation (by index in the self.dag_nodes array) and
        updates the graph accordingly.
        :param n: int, the id of the operation to be popped
        :return: None
        :raises: IndexError, if the all predecessors of n are not already popped
        """
        if self.__indegree[n] != 0:
            raise IndexError('Cannot pop a node which is not a leaf node from the Circuit DAG')
        # Remove from the DAG
        for v in self.dag.successors(self.dag_nodes[n]):
            idx = self.__node_to_index[v]
            self.__indegree[idx] -= 1
            if self.__indegree[idx] == 0:
                self.__leaf_queue.add(idx)
        self.__leaf_queue.remove(n)
        # Push the qubit progress array
        qubits_processed = self.get_gate_operands(n)
        for qubit in qubits_processed:
            self.__qubit_progress[qubit] += 1
            if self.__qubit_progress[qubit] >= len(self.__qubit_operations[qubit]):
                self.__qubit_progress[qubit] = -1

    @property
    def leaf_operations(self):
        """
        Returns the set of operations which can be executed without dependencies
        (does not check for hardware specification, only in the logical circuit)
        :return: set, indices of leaf operations (as in self.dag_nodes)
        """
        return self.__leaf_queue.copy()

    @property
    def gate_operands(self):
        """
        Converts the gate operations into a list of indices of qubits they operate on
        :return: list of tuples, each tuple is the operands for the i-th gate
        """
        # Making a list of tuples [g_i = (i, j, ...)] where i, j, ... are
        # the qubits operated on in the i-th gate, gates in node order
        operations = []
        for node in self.dag_nodes:
            operation: cirq.Operation = node.val
            operations.append(tuple(map(lambda x: self.__qubit_to_index[x], operation.qubits)))
        return operations

    def get_gate_operands(self, n):
        """
        Gets the indices of operands of a specific gate
        :param n: index of the gate being queried
        :return: tuple, the operands for the i-th gate
        """
        operation: cirq.Operation = self.dag_nodes[n].val
        return tuple(map(lambda x: self.__qubit_to_index[x], operation.qubits))

    @property
    def leaf_neighbors(self):
        """
        Get the list of next operation neighbor qubits for each qubit
        :return: list, of length num_qubits, next operations qubit or -1
        """
        return np.array([
            self.__qubit_operations[bit][pos] if pos != -1 else -1
            for bit, pos in enumerate(self.__qubit_progress)
        ])


def dag_from_qasm(filename):
    """
    Loads a QASM file and results the circuit dag from it
    :param filename: str, path to the file
    :return: cirq.CircuitDag(), Dependency graph for gates in circuit
    """
    return cirq.CircuitDag.from_circuit(circuit_from_qasm(filename))


def circuit_from_qasm(filename):
    """
    Loads a QASM file and results the circuit dag from it
    :param filename: str, path to the file
    :return: cirq.CircuitDag(), Dependency graph for gates in circuit
    """
    from cirq.contrib import qasm_import
    qasm = open(filename, 'r').read()
    circuit = cirq.Circuit(qasm_import.circuit_from_qasm(qasm))
    return circuit


if __name__ == '__main__':
    data = circuit_from_qasm('test/circuit_qasm/test.qasm')
    print(data)
    c = CircuitState(circuit=data)
    c.draw_circuit_graph()
