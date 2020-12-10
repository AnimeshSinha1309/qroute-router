import numpy as np
import cirq

from qroute.environment.circuits import CircuitRepDQN
from qroute.environment.device import DeviceTopology


def validate_solution(circuit: CircuitRepDQN, output: list, initial_locations: np.ndarray, device: DeviceTopology):
    """

    :param circuit: CircuitRep object, defines the input problem
    :param output: list, Gates in the array as (n1, n2, type)
    :param initial_locations: The starting node to qubit mapping
    :param device: The device we are compiling the circuit on
    :return:
    """
    output_circuit = cirq.Circuit()
    output_qubits = cirq.LineQubit.range(len(initial_locations))

    qubit_locations = initial_locations
    circuit_progress = np.zeros(len(circuit), dtype=np.int)
    for gate in output:
        n1, n2, t = gate
        q1, q2 = qubit_locations[n1], qubit_locations[n2]
        assert device.is_adjacent((n1, n2)), "Cannot Schedule gate on non-adjacent bits"
        if t == 'swap':
            qubit_locations[n1], qubit_locations[n2] = q2, q1
            output_circuit.append(cirq.SWAP(output_qubits[q1], output_qubits[q2]))
        elif t == 'cnot':
            assert circuit.circuit[q1][circuit_progress[q1]] == q2, "Unexpected CNOT scheduled"
            assert circuit.circuit[q2][circuit_progress[q2]] == q1, "Unexpected CNOT scheduled"
            circuit_progress[qubit_locations[n1]] += 1
            circuit_progress[qubit_locations[n2]] += 1
            output_circuit.append(cirq.CX(output_qubits[q1], output_qubits[q2]))
    for idx, progress in enumerate(circuit_progress):
        assert progress == len(circuit.circuit[idx]), "Operations were not completed"

    # Print to see what your circuit looks like
    # print(circuit.cirq)
    # print(output_circuit)
    return len(output_circuit.moments)
