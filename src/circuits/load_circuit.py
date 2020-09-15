import cirq
from cirq.contrib import qasm_import


def from_qasm(filename):
    qasm = open(filename, 'r').read()
    circuit = cirq.Circuit(qasm_import.circuit_from_qasm(qasm))
    return circuit


if __name__ == "__main__":
    print(from_qasm('src/architectures/device_qasm/radd_250.qasm'))
