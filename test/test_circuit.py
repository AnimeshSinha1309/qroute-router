import circuits


def test_dag_from_qasm():
    c = circuits.load_circuit.dag_from_qasm('circuit_qasm/3_17_13_onlyCX.qasm')
    assert c.number_of_nodes() == 17
    assert c.number_of_edges() == 136
