# import os
import logging

# import wandb

import qroute

logging.basicConfig(level=logging.DEBUG)

# os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
# wandb.init(project='qroute-rl', name='dqn-basic-1', save_code=False)


if __name__ == '__main__':
    _device = qroute.environment.device.GridComputerDevice(2, 2)
    _cirq = qroute.environment.circuits.circuit_generated_full_layer(len(_device), 5)
    _circuit = qroute.environment.circuits.CircuitRepDQN(_cirq)
    assert len(_circuit) == len(_device), "All qubits on target hardware need to be used once #FIXME"
    _model = qroute.models.actor_critic.ActorCriticAgent(_device)
    _agent = qroute.algorithms.actanneal.AnnealerAct(_model, _device)
    _memory = qroute.memory.list.MemorySimple(500)
    for e in range(300):
        qroute.engine.train_step(_agent, _device, _circuit, _memory, training_steps=500, episode_id=e+1)
