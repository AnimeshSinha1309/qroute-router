import pytest
import qroute


def test_execute():
    """
    Tests if the state has correct implementation for circuit execute all and execute swap given the
    device constraints
    """
    c = qroute.environment.circuits.CircuitState(
        circuit=qroute.environment.circuits.circuit_from_qasm('test/circuit_qasm/test.qasm'))
    d = qroute.environment.device.IBMqx5Device()
    s = qroute.environment.state.State(circuit=c, device=d)
    assert s.execute_executable() == {(2, 3), (1,)}
    assert s.mapping == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None]
    assert s.execute_all(swaps=[(7, 8)]) == {(7, 8)}
    pytest.raises(ValueError, lambda: s.execute_all(swaps=[(2, 4)]))
    assert s.mapping == [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, None, None, None, None, None, None]
    assert s.execute_all(swaps=[(5, 12), (9, 10), (0, 15), (2, 3)]) == {(2, 4)}
    assert s.mapping == [None, 1, 3, 2, 4, None, 6, 8, 7, None, 9, None, 5, None, None, 0]
    pytest.raises(ValueError, lambda: s.execute_all(swaps=[(10, 13)]))
    assert s.execute_all(swaps=[(12, 13), (10, 11), (7, 8)]) == {(6, 7)}
    assert s.mapping == [None, 1, 3, 2, 4, None, 6, 7, 8, None, None, 9, None, 5, None, 0]
    assert s.execute_all(swaps=[(13, 14), (11, 12)]) == {(0, 5)}
    assert s.mapping == [None, 1, 3, 2, 4, None, 6, 7, 8, None, None, None, 9, None, 5, 0]
    assert s.execute_all(swaps=[(12, 13)]) == {(2, 5)}
    assert s.mapping == [None, 1, 3, 2, 4, None, 6, 7, 8, None, None, None, None, 9, 5, 0]
    assert s.execute_all(swaps=[(1, 2)]) == set()
    assert s.mapping == [None, 3, 1, 2, 4, None, 6, 7, 8, None, None, None, None, 9, 5, 0]
    assert s.execute_all(swaps=[(2, 3), (13, 14)]) == {(0, 9)}
    assert s.mapping == [None, 3, 2, 1, 4, None, 6, 7, 8, None, None, None, None, 5, 9, 0]
    assert s.execute_all(swaps=[]) == {(1, 9)}
    assert s.mapping == [None, 3, 2, 1, 4, None, 6, 7, 8, None, None, None, None, 5, 9, 0]
    assert s.circuit.leaf_operations == set()
