class State:
    """
    The state is maintained as a graph of circuit and the mapping of logical to physical qubits

    Notes
    -----
    * The term "leaf" is used to refer to the layer of operations LC_0 which will be executed next.
      Leaf neighbors however, may encompass more than LC_0 if the node being queried has no operations in the
      leaf layer, then the leaf neighbor is the first layer where it's used in an operation.
    """

    def __init__(self):
        pass

    def __copy__(self):
        pass

    @property
    def circuit_dag(self):
        return None

    @property
    def leaf_neighbors(self):
        return None

    @property
    def leaf_ops(self):
        return None
