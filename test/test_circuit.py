import qroute as q


def test_dag_from_qasm():
    c = q.circuits.load_circuit.dag_from_qasm('test/circuit_qasm/3_17_13_onlyCX.qasm')
    assert c.number_of_nodes() == 17
    assert c.number_of_edges() == 136


def test_generate_random_cirq_circuit():
    c = q.circuits.random_circuit.generate_random_cirq_circuit(5, 20, 0.8)
    assert len(list(c.all_qubits())) == 5
    assert len(list(c.all_operations())) >= 20  # All operations = 2-Qubits-Op + 3 * (1-Qubit-Op) >= 20
