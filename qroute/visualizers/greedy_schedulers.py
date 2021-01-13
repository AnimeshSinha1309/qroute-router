"""
External routing software using Cirq greedy routing
"""

import networkx as nx
import cirq.contrib.routing.greedy


def cirq_routing(circuit, device):
    """
    Solves the qubit routing problem using Cirq greedy routing
    :param circuit: the input logical circuit to route
    :param device: the device we are trying to compile to
    :return: swap circuit, like an actual circuit but with swap operations inserted with logical nomenclature
    """
    device_graph = nx.Graph()
    for edge in device.edges:
        device_graph.add_edges_from([(cirq.LineQubit(edge[0]), cirq.LineQubit(edge[1]))])
    swap_network = cirq.contrib.routing.greedy.route_circuit_greedily(circuit.cirq, device_graph)
    return swap_network
