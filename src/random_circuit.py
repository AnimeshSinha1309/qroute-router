import numpy as np
import cirq

# Defining the hyper-parameters for the generator
NUM_QUBIT = 20
TWO_QUBIT_PROB = 0.4
OPERATIONS = 100

free_circuit = cirq.Circuit()
qubits = cirq.LineQubit.range(NUM_QUBIT)

for i in range(OPERATIONS):
    two_qubit = np.random.choice([True, False], p=[TWO_QUBIT_PROB, 1.0 - TWO_QUBIT_PROB])
    if two_qubit:
        indices = np.random.choice(range(NUM_QUBIT), size=2, replace=False)
        q1, q2 = qubits[indices[0]], qubits[indices[1]]
        free_circuit.append(cirq.CNOT(q1, q2))
    else:
        indices = np.random.choice(range(NUM_QUBIT), size=1)
        q = qubits[indices[0]]
        free_circuit.append(cirq.X(q))
        free_circuit.append(cirq.Y(q))
        free_circuit.append(cirq.X(q))

print(free_circuit)
