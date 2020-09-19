import cirq
import networkx as nx
import numpy as np


class DeviceTopology(cirq.Device):

    def __init__(self):
        self.graph = nx.DiGraph()

    def __get_distance_matrix(self):
        """
        Uses the Floyd-Warshall algorithm to compute the distance between all pairs of qubits
        :return: matrix of integers of size (n,n), (i,j) contains distance of i to j
        :except: AttributeError if graph is not initialized (or logical error if edges not loaded)
        """
        mat = np.full(fill_value=9999999999, shape=(self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        for source, dest in self.graph.edges:
            mat[source][dest] = 0
        for k in range(self.graph.number_of_nodes()):
            for i in range(self.graph.number_of_nodes()):
                for j in range(self.graph.number_of_nodes()):
                    mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
        return mat

    def validate_operation(self, operation):
        if not isinstance(operation, cirq.GateOperation):
            raise ValueError('{!r} is not a supported operation'.format(operation))
        if not isinstance(operation.gate, (cirq.CZPowGate, cirq.XPowGate, cirq.PhasedXPowGate, cirq.YPowGate)):
            raise ValueError('{!r} is not a supported gate'.format(operation.gate))
        if len(operation.qubits) == 2:
            p, q = operation.qubits
            if not self.graph.has_edge(p, q):
                raise ValueError('Non-local interaction: {}'.format(repr(operation)))

    def validate_circuit(self, circuit):
        for moment in circuit:
            for operation in moment.operations:
                self.validate_operation(operation)

    def draw_architecture_graph(self):
        nx.draw(self.graph, with_labels=True)


class IBMqx3Device(DeviceTopology):

    def __init__(self):
        super(IBMqx3Device, self).__init__()
        self.graph.add_nodes_from(list(range(16)))
        self.graph.add_edges_from([(1, 2), (1, 0), (2, 3), (3, 4), (3, 14), (5, 4), (6, 5), (6, 11), (6, 7), (7, 10),
                                   (8, 7), (9, 8), (9, 10), (11, 10), (12, 5), (12, 11), (12, 13), (13, 14), (13, 4),
                                   (15, 14), (15, 2), (15, 0)])
        self.distances = self.__get_distance_matrix()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    device = IBMqx3Device()
    device.draw_architecture_graph()
    plt.show()
