# import os
import logging

# import wandb

import qroute

logging.basicConfig(level=logging.DEBUG)

# os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
# wandb.init(project='qroute-rl', name='dqn-basic-1', save_code=False)


if __name__ == '__main__':
    device = qroute.environment.device.GridComputerDevice(6, 6)
    cirq = qroute.environment.circuits.circuit_generated_randomly(len(device), 100)
    circuit = qroute.environment.circuits.CircuitRepDQN(cirq, len(device))
    model = qroute.models.graph_dual.GraphDualModel(device, True)
    agent = qroute.algorithms.deepmcts.MCTSAgent(model, device)
    memory = qroute.memory.list.MemorySimple(500)

    for e in range(300):
        qroute.engine.train_step(agent, device, circuit, memory, training_steps=100, episode_id=e+1)
