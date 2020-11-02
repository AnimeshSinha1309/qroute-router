from environment.device import IBMqx5Device
from environment.circuits import CircuitState, circuit_from_qasm
from environment.state import State
from models.graph_conv import GraphConvNetwork

device = IBMqx5Device()
device.draw_architecture_graph()
circuit = CircuitState(circuit=circuit_from_qasm('test/circuit_qasm/test.qasm'))
circuit.draw_circuit_graph()
state = State(device, circuit)

net = GraphConvNetwork(state.circuit)
