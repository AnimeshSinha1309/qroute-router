import cirq
from cirq.contrib import qasm_import


def dag_from_qasm(filename):
    """
    Loads a QASM file and results the circuit dag from it
    :param filename: str, path to the file
    :return: cirq.CircuitDag(), Dependency graph for gates in circuit
    """
    qasm = open(filename, 'r').read()
    circuit = cirq.Circuit(qasm_import.circuit_from_qasm(qasm))
    return cirq.CircuitDag.from_circuit(circuit)
