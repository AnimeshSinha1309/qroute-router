#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit
from qiskit.compiler.transpile import transpile

from pytket.qasm import circuit_from_qasm, circuit_from_qasm_str
from pytket.routing import Architecture, route
from pytket.transform import Transform
from pytket.qiskit import tk_to_qiskit

import cirq

path = '../../test/circuit_qasm'

def circs_from_direc(path):
    """ 
        Searches for circuits in the test folder. 
        Args: path -> str // path for the test folder
        Returns: test_circs_path -> list // paths to all the qasm files
    """

    from os import listdir
    from os.path import isfile, join
    
    test_circs_path = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return test_circs_path

def qiskit_routing(obj, device):
    """ 
        Routing using qiskit
        Args: obj -> str/Cirq.circuit // Either qasm filepath or Cirq's circuit object
              device -> Hardware device object // Defines topology for hardware specific routing   
        Return: routing -> dict // Dict of circuit depth according to four qiskit routing strategies
    """

    if isinstance(obj, cirq.circuits.circuit.Circuit):
        qcircuit = QuantumCircuit.from_qasm_str(obj.to_qasm(header=''))
    else:
        qcircuit = QuantumCircuit.from_qasm_file(str(obj))
        
    if qcircuit.num_qubits <= device.nodes:
        coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
        routing = {'basic':0, 'lookahead':0, 'stochastic':0, 'sabre':0}
        for rt in routing.keys():
            trcircuit = transpile(qcircuit, coupling_map=coupling_map, routing_method=rt, optimization_level=0)
            routing.update({rt:trcircuit.depth()})
        return routing
    else:
        return -1

def tket_routing(obj, device):
    """ 
        Routing using qiskit
        Args: obj -> str/Cirq.circuit // Either qasm filepath or Cirq's circuit object
              device -> Hardware device object // Defines topology for hardware specific routing   
        Return: depth -> int // Circuit depth according to tket routing strategies
    """

    if isinstance(obj, cirq.circuits.circuit.Circuit):
        qcircuit = circuit_from_qasm_str(obj.to_qasm(header=''))
    else:
        qcircuit = circuit_from_qasm(str(obj))
        
    if qcircuit.n_qubits <= device.nodes:
        architecture = Architecture(list(map(list, device.edges)) + list(map(list, map(reversed, device.edges))))
        rcircuit = route(circuit=qcircuit, architecture=architecture)
        Transform.DecomposeBRIDGE().apply(rcircuit)
        Transform.RemoveRedundancies().apply(rcircuit)
        return rcircuit.depth()
    else:
        return -1

