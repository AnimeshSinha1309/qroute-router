import numpy as np
import cirq


class CircuitRepDQN:
    """
    Keeps a global usable representation of a logical circuit
    """

    def __init__(self, circuit: cirq.Circuit):
        """
        Takes in a circuit (logical circuit - LC) and initializes the state object
        which will maintain the leaf nodes (operations which can be executed without dependencies)
        and the next node that each element wants to qubit with (for swapping heuristics)
        :param circuit: cirq.Circuit, The input logical circuit

        >>> test_c = CircuitRepDQN(circuit_from_qasm('test/circuit_qasm/test.qasm'))
        """
        self.qubits = list(circuit.all_qubits())
        operators = circuit.all_operations()
        self.circuit: list = [[]] * len(self.qubits)
        for operator in operators:
            if len(operator.qubits) == 1:
                continue
            elif len(operator.qubits) == 2:
                q1, q2 = self.qubits.index(operator.qubits[0]), self.qubits.index(operator.qubits[1])
                self.circuit[q1].append(q2)
                self.circuit[q2].append(q1)
            else:
                raise ValueError('3 qubit primitives are not valid gates in the circuit')
        self.progress = np.zeros(len(self.qubits))

    def __getitem__(self, item: int):
        return self.circuit[item]

    def __len__(self):
        return len(self.qubits)


def circuit_from_qasm(filename):
    """
    Loads a QASM file and returns the circuit from it
    :param filename: str, path to the file
    :return: cirq.CircuitDag(), Dependency graph for gates in circuit
    """
    from cirq.contrib import qasm_import
    qasm = open(filename, 'r').read()
    circuit = cirq.Circuit(qasm_import.circuit_from_qasm(qasm))
    return circuit


def circuit_generated_randomly(num_qubits=20, num_cx=100):
    """
    Returns one cirq circuit randomly initialized
    :param num_qubits: int, number of qubits in the random circuit
    :param num_cx: int, number of random Controlled X gates
    :return: cirq.Circuit(), output random circuit
    """
    free_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(num_qubits)

    for _ in range(num_cx):
        indices = np.random.choice(range(num_qubits), size=2, replace=False)
        q1, q2 = qubits[indices[0]], qubits[indices[1]]
        free_circuit.append(cirq.CX(q1, q2))
    return free_circuit


def qasm_generated_randomly(num_qubits, num_cx, num_files, path, start_num=0):
    """
    Writes several QASM files with random circuits consisting of Controlled X gates
    :param num_qubits: int, number of qubits in the random circuit
    :param num_cx: int, number of random Controlled X gates
    :param num_files: int, number of independent files we are writing
    :param path: str, directory to which the files should be written to
    :param start_num: int, offset in the filenames being written
    :return: None
    """
    for i in range(num_files):
        full_path = path + str(start_num + i) + '.qasm'
        print('\rGenerating %d of %d file' % (i + 1, num_files), end='')
        with open(full_path, 'w') as f:
            f.write('OPENQASM 2.0;\ninclude "qelib1.inc";')
            f.write('\nqreg q[' + str(num_qubits) + '];')
            f.write('\ncreg c[' + str(num_qubits) + '];')
            for _ in range(num_cx):
                cx1 = str(np.random.randint(num_qubits))
                cx2 = str(np.random.randint(num_qubits))
                while cx2 == cx1:
                    cx2 = str(np.random.randint(num_qubits))
                f.write('\ncx q[' + cx1 + '],q[' + cx2 + '];')
