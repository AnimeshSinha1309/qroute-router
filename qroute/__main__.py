import qroute


device = qroute.IBMqx5Device()
device.draw_architecture_graph()
circuit = CircuitState(circuit=circuit_from_qasm('test/circuit_qasm/test.qasm'))
circuit.draw_circuit_graph()
state = State(device, circuit)

net = GraphConvNetwork(state.circuit)
