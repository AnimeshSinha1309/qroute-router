import cirq
import networkx as nx
import numpy as np


class CircuitState:

    def __init__(self, circuit: cirq.Circuit):
        self.circuit = circuit
        # TODO: Make this DAG object global and shared between all instances, immutable
        self.dag = cirq.CircuitDag.from_circuit(self.circuit)
        self.dag_nodes = list(self.dag.nodes)
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

    def draw_circuit_graph(self):
        nx.draw(self.dag, with_labels=True)

    def pop(self, n):
        for v in self.dag.successors(self.dag_nodes[n]):
            idx = self.__node_to_index[v]
            self.__indegree[idx] -= 1
            if self.__indegree[idx] == 0:
                self.__leaf_queue.add(idx)
        self.__leaf_queue.remove(n)

    @property
    def leaf_nodes(self):
        return self.__leaf_queue.copy()

    @property
    def operations(self):
        # Making a list of tuples [g_i = (i, j, ...)] where i, j, ... are
        # the qubits operated on in the i-th gate, gates in node order
        qubits = list(sorted(self.circuit.all_qubits()))
        operations = []
        for node in self.dag_nodes:
            operation: cirq.Operation = node.val
            operations.append(tuple(map(qubits.index, operation.qubits)))
        return operations


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


c = CircuitState(circuit=circuit_from_qasm('test/circuit_qasm/test.qasm'))
