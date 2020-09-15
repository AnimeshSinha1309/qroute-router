import numpy as np
import cirq


def generate_random_cirq_circuit(NUM_QUBIT = 20, NUM_CX = 100, TWO_QUBIT_PROB = 0.4):
    """
    Returns one cirq circuit randomly initialized

    Params
    ======
    NUM_QUBIT: int, number of qubits in the random circuit
    NUM_CX: int, number of random Controlled X gates
    TWO_QUBIT_PROB: float, probability of two qubit gates vs. one qubit

    Returns
    =======
    free_circuit: cirq.Circuit(), output random circuit
    """
    free_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(NUM_QUBIT)

    for _ in range(NUM_CX):
        two_qubit = np.random.choice([True, False], p=[TWO_QUBIT_PROB, 1.0 - TWO_QUBIT_PROB])
        if two_qubit:
            indices = np.random.choice(range(NUM_QUBIT), size=2, replace=False)
            q1, q2 = qubits[indices[0]], qubits[indices[1]]
            free_circuit.append(cirq.CX(q1, q2))
        else:
            indices = np.random.choice(range(NUM_QUBIT), size=1)
            q = qubits[indices[0]]
            free_circuit.append(cirq.X(q))
            free_circuit.append(cirq.Y(q))
            free_circuit.append(cirq.X(q))
    return free_circuit


def write_random_qasm_file(NUM_QUBIT, NUM_CX, num_files, path, start_num=0):
    """
    Writes several QASM files with random circuits consisting of Controlled X gates

    Params
    ======
    NUM_QUBIT: int, number of qubits in the random circuit
    NUM_CX: int, number of random Controlled X gates
    num_files: int, number of independent files we are writing
    path: str, directory to which the files should be written to
    start_num: int, offset in the filenames being written

    Outputs
    =======
    num_files QASM files with random circuits
    """
    for i in range(num_files):
        full_path = path + str(start_num + i) + '.qasm'
        print('\rGenerating %d of %d file'%(i + 1, num_files), end='')
        with open(full_path, 'w') as f:
            f.write('OPENQASM 2.0;\ninclude "qelib1.inc";')
            f.write('\nqreg q[' + str(NUM_QUBIT) + '];')
            f.write('\ncreg c[' + str(NUM_QUBIT) + '];')
            for _ in range(NUM_CX):
                cx1 = str(np.random.randint(NUM_QUBIT))
                cx2 = str(np.random.randint(NUM_QUBIT))
                while cx2 == cx1: cx2 = str(np.random.randint(NUM_QUBIT))
                f.write('\ncx q[' + cx1 + '],q[' + cx2 + '];')
