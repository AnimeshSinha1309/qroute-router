from qroute import environment as env


def test_circuit_pop():
    c = env.circuits.CircuitState(circuit=env.circuits.circuit_from_qasm('test/circuit_qasm/test.qasm'))
    assert c.leaf_nodes == {0, 1, 2, 3}
    c.pop(1)
    assert c.leaf_nodes == {0, 2, 3, 4}
    c.pop(0)
    assert c.leaf_nodes == {2, 3, 4}
    c.pop(4)
    assert c.leaf_nodes == {2, 3, 8}
    c.pop(8)
    assert c.leaf_nodes == {2, 3}
    c.pop(2)
    assert c.leaf_nodes == {3, 5}
    c.pop(3)
    assert c.leaf_nodes == {5, 6}
    c.pop(6)
    assert c.leaf_nodes == {5}
    c.pop(5)
    assert c.leaf_nodes == {7}
    c.pop(7)
    assert c.leaf_nodes == set()


def test_dag_from_qasm():
    c = env.circuits.dag_from_qasm('test/circuit_qasm/3_17_13_onlyCX.qasm')
    assert c.number_of_nodes() == 17
    assert c.number_of_edges() == 136


def test_generate_random_cirq_circuit():
    c = env.generate.generate_random_cirq_circuit(5, 20, 0.8)
    assert len(list(c.all_qubits())) == 5
    assert len(list(c.all_operations())) >= 20  # All operations = 2-Qubits-Op + 3 * (1-Qubit-Op) >= 20
