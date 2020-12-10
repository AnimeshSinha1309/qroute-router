import numpy as np

from qroute.environment.state import CircuitStateDQN


def heuristic_action(state: CircuitStateDQN):
    gates = state.next_gates()
    device = state.device

    gates_to_force = list(filter(lambda gate: device.distances[gate[0], gate[1]] == 2, gates))
    if len(gates_to_force) == 0:
        return None
    np.random.shuffle(gates_to_force)

    action = np.array([0] * len(device.edges))  # an action representing an empty layer of swaps
    available_edges = set(filter(lambda e: e[0] not in state.protected_nodes and
                                           e[1] not in state.protected_nodes, device.edges))
    edge_index_map = {edge: index for index, edge in enumerate([(n1, n2) for (n1, n2) in device.edges])}

    for n1, n2 in gates_to_force:
        n1_neighbours = np.where(device.distances[n1] == 1)[0]
        n2_neighbours = np.where(device.distances[n2] == 1)[0]
        intermediate_nodes = np.intersect1d(n1_neighbours, n2_neighbours)
        np.random.shuffle(intermediate_nodes)

        for n3 in intermediate_nodes:
            possible_edges = [((n1, n3) if n1 < n3 else (n3, n1)),
                              ((n2, n3) if n2 < n3 else (n3, n2))]
            np.random.shuffle(possible_edges)

            for edge_to_swap in possible_edges:
                if edge_to_swap not in available_edges:
                    continue

                action[edge_index_map[edge_to_swap]] = 1
                fixed_node = n1 if n1 not in edge_to_swap else n2
                edges = [(m1, m2) for (m1, m2) in device.edges]

                for edge in edges:
                    if edge_to_swap[0] in edge or edge_to_swap[1] in edge or fixed_node in edge:
                        if edge in available_edges:
                            available_edges.remove(edge)

    return action
