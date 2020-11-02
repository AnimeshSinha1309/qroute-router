import dgl
import dgl.nn
from environment.circuits import CircuitState


class GraphConvNetwork:

    def __init__(self, circuit: CircuitState):
        """
        Initializes the graph network as a torch module and makes the
        architecture and the graph.
        :param circuit:
        """
        super(GraphConvNetwork, self).__init__()
        self.circuit = circuit.dag
        self.graph = dgl.DGLGraph().from_networkx(self.circuit)
        self.model = dgl.nn.GraphConv(4, len(circuit))
