import cirq
import collections
import numpy as np


class CircuitState:

    def __init__(self, circuit: cirq.Circuit):
        self.circuit = circuit
        self.dag = cirq.CircuitDag.from_circuit(self.circuit)
        self.qubits = list(sorted(self.circuit.all_qubits()))
        self.dag_nodes = list(self.dag.nodes)
        self.__node_to_index = {node: idx for idx, node in enumerate(self.dag_nodes)}
        # Making a list of tuples [g_i = (i, j, ...)] where i, j, ... are the qubits operated on
        self.operations = []
        for node in self.dag.nodes:
            operation: cirq.Operation = node.val
            self.operations.append(tuple(map(self.qubits.index, operation.qubits)))
        # Make the queue of leaf nodes
        self.__queue = collections.deque()
        self.__indegree = np.zeros(shape=self.dag.number_of_nodes())
        for u in self.dag_nodes:
            for v in self.dag.successors(u):
                self.__indegree[self.__node_to_index[v]] += 1
        for idx, indegree in enumerate(self.__indegree):
            if self.__indegree[idx] == 0:
                self.__queue.append(idx)
        print(self.__indegree)
        print(self.__queue)


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


c = CircuitState(circuit=circuit_from_qasm('test/circuit_qasm/3_17_13.qasm'))
