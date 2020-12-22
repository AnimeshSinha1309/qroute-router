import numpy as np
import cirq

from qroute.environment.circuits import CircuitRepDQN
from qroute.environment.env import Moment
from qroute.environment.device import DeviceTopology


def validate_solution(circuit: CircuitRepDQN, output: list, initial_locations: np.ndarray, device: DeviceTopology,
                      verbose=True):
    """

    :param circuit: CircuitRep object, defines the input problem
    :param output: list, Gates in the array as (n1, n2, type)
    :param initial_locations: The starting node to qubit mapping
    :param device: The device we are compiling the circuit on
    :param verbose: bool, whether to print the circuits
    :return:
    """
    output_circuit = cirq.Circuit()
    output_qubits = cirq.LineQubit.range(len(initial_locations))

    qubit_locations = initial_locations
    circuit_progress = np.zeros(len(circuit), dtype=np.int)

    moment: Moment
    for moment in output:
        for n1, n2 in moment.cnots:
            q1, q2 = qubit_locations[n1], qubit_locations[n2]
            assert device.is_adjacent((n1, n2)), "Cannot Schedule gate on non-adjacent bits"
            assert circuit.circuit[q1][circuit_progress[q1]] == q2, "Unexpected CNOT scheduled"
            assert circuit.circuit[q2][circuit_progress[q2]] == q1, "Unexpected CNOT scheduled"
            circuit_progress[qubit_locations[n1]] += 1
            circuit_progress[qubit_locations[n2]] += 1
            output_circuit.append(cirq.CX(output_qubits[q1], output_qubits[q2]))
        for n1, n2 in moment.swaps:
            q1, q2 = qubit_locations[n1], qubit_locations[n2]
            assert device.is_adjacent((n1, n2)), "Cannot Schedule gate on non-adjacent bits"
            qubit_locations[n1], qubit_locations[n2] = q2, q1
            output_circuit.append(cirq.SWAP(output_qubits[q1], output_qubits[q2]))
    for idx, progress in enumerate(circuit_progress):
        assert progress == len(circuit.circuit[idx]), "Operations were not completed"

    # Print to see what your circuit looks like
    if verbose:
        print()
        print("Initial State:", initial_locations)
        print("Input Circuit\n", circuit.cirq)
        print("Output Circuit\n", output_circuit)
    return output_circuit


def check_valid_solution(solution, device):
    """
    Checks if a solution is valid, i.e. does not use one node twice
    :param solution: list, boolean array of swaps, the solution to check
    :param device: DeviceTopology, the device to check on
    :raises: RuntimeError if the solution is invalid
    """
    if not np.any(solution):
        return
    swap_edge_indices = np.where(np.array(solution) == 1)[0]
    swap_edges = [device.edges[index] for index in swap_edge_indices]
    swap_nodes = [node for edge in swap_edges for node in edge]

    # return False if repeated swap nodes
    seen = set()
    for node in swap_nodes:
        if node in seen:
            raise RuntimeError('Solution is not safe: A node has several ops - %s' % str(swap_edges))
        seen.add(node)
