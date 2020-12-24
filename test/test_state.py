import numpy as np

import qroute


def test_cnot_execution():
    cirq_circuit = qroute.environment.circuits.circuit_generated_full_layer(4, 1)
    circuit = qroute.environment.circuits.CircuitRepDQN(cirq_circuit)
    device = qroute.environment.device.GridComputerDevice(2, 2)
    # Testing complete execution
    state = qroute.environment.state.CircuitStateDQN(circuit, device, node_to_qubit=np.arange(4))
    gates_executed = state.execute_cnot()
    for n1, n2 in [(0, 1), (2, 3)]:
        assert (n1, n2) in gates_executed or (n2, n1) in gates_executed, "Right gates were not executed"
    assert len(gates_executed) == 2, "Not the right number of gates executed"
    # Testing null execution
    state = qroute.environment.state.CircuitStateDQN(circuit, device, node_to_qubit=np.array([0, 3, 2, 1]))
    gates_executed = state.execute_cnot()
    assert len(gates_executed) == 0, "Not the right number of gates executed"


def test_swap_execution():
    cirq_circuit = qroute.environment.circuits.circuit_generated_full_layer(4, 1)
    circuit = qroute.environment.circuits.CircuitRepDQN(cirq_circuit)
    device = qroute.environment.device.GridComputerDevice(2, 2)
    # Testing null execution
    state = qroute.environment.state.CircuitStateDQN(circuit, device, node_to_qubit=np.array([0, 3, 2, 1]))
    gates_executed = state.execute_cnot()
    assert len(gates_executed) == 0, "Not the right number of gates executed"
    # Performing the swaps
    action = np.array([True if edge in [(1, 3), (3, 1)] else False for edge in device.edges])
    state.execute_swap(action)
    # Testing complete execution
    gates_executed = state.execute_cnot()
    for n1, n2 in [(0, 1), (2, 3)]:
        assert (n1, n2) in gates_executed or (n2, n1) in gates_executed, "Right gates were not executed"
    assert len(gates_executed) == 2, "Not the right number of gates executed"
