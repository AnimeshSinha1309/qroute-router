import numpy as np

from qroute import environment as env


def test_circuit_pop():
    """
    Tests if the init and pop operations of the circuit class as well as
    the gate_operands, leaf_operands, leaf_neighbors and other properties are working

    note: The explanations for the test set are available in the diagram drown here:
          https://github.com/AnimeshSinha1309/quantum-rl/issues/5
    """
    c = env.circuits.CircuitState(circuit=env.circuits.circuit_from_qasm('test/circuit_qasm/test.qasm'))
    gate_to_operands = {idx: val for idx, val in enumerate(c.gate_operands)}
    operands_to_gate = {val: idx for idx, val in enumerate(c.gate_operands)}
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 3), (0, 5), (1,), (7, 8)}
    assert np.all(c.leaf_neighbors == np.array([5, 1, 3, 2, 2, 0, 7, 8, 7, 0]))
    c.pop(operands_to_gate[7, 8])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 3), (0, 5), (1,), (6, 7)}
    assert np.all(c.leaf_neighbors == np.array([5, 1, 3, 2, 2, 0, 7, 6, -1, 0]))
    c.pop(operands_to_gate[0, 5])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 3), (0, 9), (1,), (6, 7)}
    assert np.all(c.leaf_neighbors == np.array([9, 1, 3, 2, 2, 2, 7, 6, -1, 0]))
    c.pop(operands_to_gate[1,])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 3), (0, 9), (6, 7)}
    assert np.all(c.leaf_neighbors == np.array([9, 9, 3, 2, 2, 2, 7, 6, -1, 0]))
    c.pop(operands_to_gate[2, 3])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 4), (0, 9), (6, 7)}
    assert np.all(c.leaf_neighbors == np.array([9, 9, 4, -1, 2, 2, 7, 6, -1, 0]))
    c.pop(operands_to_gate[2, 4])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 5), (0, 9), (6, 7)}
    assert np.all(c.leaf_neighbors == np.array([9, 9, 5, -1, -1, 2, 7, 6, -1, 0]))
    c.pop(operands_to_gate[6, 7])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(2, 5), (0, 9)}
    assert np.all(c.leaf_neighbors == np.array([9, 9, 5, -1, -1, 2, -1, -1, -1, 0]))
    c.pop(operands_to_gate[2, 5])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(0, 9)}
    assert np.all(c.leaf_neighbors == np.array([9, 9, -1, -1, -1, -1, -1, -1, -1, 0]))
    c.pop(operands_to_gate[0, 9])
    assert set(map(lambda x: gate_to_operands[x], c.leaf_operations)) == {(1, 9)}
    assert np.all(c.leaf_neighbors == np.array([-1, 9, -1, -1, -1, -1, -1, -1, -1, 1]))
    c.pop(operands_to_gate[1, 9])
    assert c.leaf_operations == set()
    assert np.all(c.leaf_neighbors == np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]))


def test_dag_from_qasm():
    """
    Basic and incomplete test for checking if a DAG is loaded correctly from QASM file
    Only checks the number of nodes and edges
    """
    c = env.circuits.dag_from_qasm('test/circuit_qasm/3_17_13_onlyCX.qasm')
    assert c.number_of_nodes() == 17
    assert c.number_of_edges() == 136


def test_generate_random_cirq_circuit():
    """
    Basic and incomplete test for checking if a DAG is correctly randomly initialized
    Only checks the number of nodes and lower bound on edges
    """
    c = env.generate.generate_random_cirq_circuit(5, 20, 0.8)
    assert len(list(c.all_qubits())) == 5
    assert len(list(c.all_operations())) >= 20  # All operations = 2-Qubits-Op + 3 * (1-Qubit-Op) >= 20
