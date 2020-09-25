import cirq
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class DeviceTopology(cirq.Device):
    """
    Defines what pairs of qubits are local and can be operated upon
    """

    def __init__(self, nodes, edges):
        """
        Creates a device with the input graph topology
        :param nodes: iterable, list of qubits, eg. [1, 2, 3, 4]
        :param edges: iterable, list of edges, eg. [(1, 2), (2, 3), (2, 4), (3, 4)]
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        self.distances = self.__get_distance_matrix(bidirectional=False)
        self.swap_dist = self.__get_distance_matrix(bidirectional=True)

    def __len__(self):
        """
        Number of qubits available in the device
        :return: int, number of qubits
        """
        return self.graph.number_of_nodes()

    @property
    def connected_qubits(self):
        """
        Get the list of all 2-tuples which are connected
        :return: list, of connected edges as 2-tuples
        """
        return self.graph.edges

    def is_adjacent(self, qubits):
        """
        Checks if 2 qubits are adjacent and can be operated upon in the hardware
        :param qubits: The list of qubits to check locality
        :return: True if the operation is local, False otherwise
        :raises: ValueError, if there are more than 2 operands in the operation.
        """
        if len(qubits) <= 1:
            return True
        elif len(qubits) == 2:
            q1, q2 = qubits
            return self.distances[q1][q2] <= 1
        else:
            raise ValueError('{!r} is multi-qubit (>2) gate'.format(qubits))

    def __get_distance_matrix(self, bidirectional=False):
        """
        Uses the Floyd-Warshall algorithm to compute the distance between all pairs of qubits
        :return: matrix of integers of size (n,n), (i,j) contains distance of i to j
        :except: AttributeError if graph is not initialized (or logical error if edges not loaded)
        """
        mat = np.full(fill_value=9999999999, shape=(self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        for bit in range(self.graph.number_of_nodes()):
            mat[bit][bit] = 0
        for source, dest in self.graph.edges:
            mat[source][dest] = 1
            if bidirectional:
                mat[dest][source] = 1
        for k in range(self.graph.number_of_nodes()):
            for i in range(self.graph.number_of_nodes()):
                for j in range(self.graph.number_of_nodes()):
                    mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
        return mat

    def validate_operation(self, operation):
        """
        Checks if an operation can be performed on the device
        :param operation: tuple, the list of qubits as operands to the current operation
        :return: None
        :raises: ValueError, if the operation is not possible on the device
        """
        if not isinstance(operation, cirq.GateOperation):
            raise ValueError('{!r} is not a supported operation'.format(operation))
        if len(operation.qubits) == 2:
            p, q = operation.qubits
            if not self.graph.has_edge(p, q):
                raise ValueError('Non-local interaction: {}'.format(repr(operation)))
        elif len(operation.qubits) > 2:
            raise ValueError('{!r} is multi-qubit (>2) gate'.format(operation))

    def validate_circuit(self, circuit):
        """
        Validates that the entire circuit is device executable
        :param circuit: cirq.Circuit, the circuit to be validated
        :return: None
        :raises: ValueError, if any operation in the circuit is an invalid (non-local) operation
        """
        for moment in circuit:
            for operation in moment.operations:
                self.validate_operation(operation)

    def draw_architecture_graph(self):
        """
        Draw the graph of the hardware topology with edges as possible operations.
        The graph is directed.
        :return: None
        """
        nx.draw(self.graph, pos=nx.circular_layout(self.graph), with_labels=False)
        plt.show()


class IBMqx5Device(DeviceTopology):
    """
    Specific device topology for the IBM QX5 device
    """

    def __init__(self):
        """
        Initialize the graph for the IBM QX5 topology
        """
        super(IBMqx5Device, self).__init__(
            nodes=list(range(16)),
            edges=[(1, 2), (1, 0), (2, 3), (3, 4), (3, 14), (5, 4), (6, 5), (6, 11),
                   (6, 7), (7, 10), (8, 7), (9, 8), (9, 10), (11, 10), (12, 5), (12, 11),
                   (12, 13), (13, 14), (13, 4), (15, 14), (15, 2), (15, 0)]
        )


if __name__ == "__main__":
    device = IBMqx5Device()
    device.draw_architecture_graph()
    print(device.distances)
