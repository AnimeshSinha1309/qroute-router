import os
import logging

# import wandb

import qroute

logging.basicConfig(level=logging.DEBUG)

# os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
# wandb.init(project='qroute-rl', name='dqn-basic-1', save_code=False)


if __name__ == '__main__':
    device = qroute.environment.device.IBMqx20TokyoDevice()
    model = qroute.models.graph_dual.GraphDualModel(device, True)
    agent = qroute.algorithms.deepmcts.MCTSAgent(model, device)
    memory = qroute.memory.list.MemorySimple(500)

    files = ["adr4_197", "radd_250", "z4_268", "sym6_145", "misex1_241",
             "rd73_252", "cycle10_2_110", "square_root_7", "sqn_258", "rd84_253", "rd84_142"]
    for e, file in enumerate(files[:1]):
        cirq = qroute.environment.circuits.circuit_from_qasm(
            os.path.join("../test/circuit_qasm", file + "_onlyCX.qasm"))
        circuit = qroute.environment.circuits.CircuitRepDQN(cirq, len(device))
        print("Working on:", len(list(cirq.all_operations())), len(list(cirq.all_qubits())))
        qroute.engine.train_step(agent, device, circuit, memory, training_steps=10000,
                                 episode_id='Circuit %s' % file)
        print("Cirq Routing Distance: ",
              len(qroute.visualizers.greedy_schedulers.cirq_routing(circuit, device).circuit.moments))
