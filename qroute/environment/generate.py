import cirq
import numpy as np


def generate_random_cirq_circuit(num_qubits=20, num_cx=100, two_qubit_prob=0.4):
    """
    Returns one cirq circuit randomly initialized
    :param num_qubits: int, number of qubits in the random circuit
    :param num_cx: int, number of random Controlled X gates
    :param two_qubit_prob: float, probability of two qubit gates vs. one qubit
    :return: cirq.Circuit(), output random circuit
    """
    free_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(num_qubits)

    for _ in range(num_cx):
        two_qubit = np.random.choice([True, False], p=[two_qubit_prob, 1.0 - two_qubit_prob])
        if two_qubit:
            indices = np.random.choice(range(num_qubits), size=2, replace=False)
            q1, q2 = qubits[indices[0]], qubits[indices[1]]
            free_circuit.append(cirq.CX(q1, q2))
        else:
            indices = np.random.choice(range(num_qubits), size=1)
            q = qubits[indices[0]]
            gate1 = cirq.rz(np.random.uniform(0.0, 2 * np.pi))
            gate2 = cirq.ry(np.random.uniform(0.0, 2 * np.pi))
            gate3 = cirq.rz(np.random.uniform(0.0, 2 * np.pi))
            free_circuit.append([gate1(q), gate2(q), gate3(q)])
    return free_circuit


def write_random_qasm_file(num_qubits, num_cx, num_files, path, start_num=0):
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
