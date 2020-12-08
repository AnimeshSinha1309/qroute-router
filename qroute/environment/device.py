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
        self.edges = edges
        self.graph = nx.Graph(nodes=nodes, edges=edges)
        self.distances = self._get_distance_matrix()

    def __len__(self):
        """
        Number of qubits available in the device
        :return: int, number of qubits
        """
        return self.graph.number_of_nodes()

    @property
    def max_distance(self):
        """
        Number of qubits available in the device
        :return: int, number of qubits
        """
        return np.max(self.distances)

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

    def _get_distance_matrix(self, bidirectional=True):
        """
        Uses the Floyd-Warshall algorithm to compute the distance between all pairs of qubits
        :param: bidirectional: bool, true if the connectivity is bidirectional, False otherwise
        :return: matrix of integers of size (n,n), (i,j) contains distance of i to j
        :except: AttributeError if graph is not initialized (or logical error if edges not loaded)

        Note that unidirectional mode may not be supported fully through the rest of the code
        """
        mat = np.full(fill_value=np.inf, shape=(len(self), len(self)))
        for bit in range(len(self)):
            mat[bit][bit] = 0
        for source, dest in self.graph.edges:
            mat[source][dest] = 1
            if bidirectional:
                mat[dest][source] = 1
        for k in range(len(self)):
            for i in range(len(self)):
                for j in range(len(self)):
                    mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
        return mat

    # Methods to check if the circuit is working on the device without violating the Topology

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

    # Some pretty printing stuff here

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
