import qroute


def test_execute():
    """
    Tests if the state has correct implementation for circuit execute all and execute swap given the
    device constraints
    """
    _device = qroute.environment.device.GridComputerDevice(4, 4)
    _circuit = qroute.environment.circuits.CircuitRepDQN(
        qroute.environment.circuits.circuit_generated_full_layer(len(_device)))
    _agent = qroute.models.double_dqn.DoubleDQNAgent(_device)
