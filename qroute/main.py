import qroute as q
import cirq

if __name__ == '__main__':
    c = q.circuits.load_circuit.dag_from_qasm('test/circuit_qasm/3_17_13_onlyCX.qasm')
    for edge in c.edges:
        print(edge[0].val, edge[1].val)
    print()

    for node in c.nodes:
        print(node.val)
