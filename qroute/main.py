import collections
# import os
import logging

import numpy as np
import tqdm
# import wandb

import qroute

logging.basicConfig(level=logging.DEBUG)

# os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
# wandb.init(project='qroute-rl', name='dqn-basic-1', save_code=False)


def train(device: qroute.environment.device.DeviceTopology,
          circuit: qroute.environment.circuits.CircuitRepDQN,
          agent,
          training_episodes=350, training_steps=500):

    num_actions_deque = collections.deque(maxlen=50)
    memory = qroute.memory.list.MemorySimple(500)

    # Training the agent
    for e in range(training_episodes):
        input_circuit = circuit
        print("Input Circuit:\n", input_circuit.cirq, flush=True)
        state = qroute.environment.state.CircuitStateDQN(input_circuit, device)
        starting_locations = np.array(state.node_to_qubit)

        progress_bar = tqdm.trange(training_steps)
        progress_bar.set_description('Episode %03d' % (e + 1))
        for time in progress_bar:
            action, _ = agent.act(state)
            next_state, reward, done, _ = qroute.environment.env.step(action, state)
            memory.store((state, reward, next_state, done))
            state = next_state

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                avg_time = np.mean(num_actions_deque)
                result_circuit = qroute.visualizers.solution_validator.validate_solution(
                    input_circuit, state.solution, starting_locations, device)
                depth = len(result_circuit.moments)
                progress_bar.set_postfix(circuit_depth=depth, num_actions=num_actions, avg_actions=avg_time)
                progress_bar.close()

                # wandb.log({'Circuit Depth': depth,
                #            'Number of Current Actions': num_actions,
                #            'Number of Average Actions': num_actions})
                print("Output Circuit:\n", result_circuit, flush=True)
                break

            agent.replay(memory)


if __name__ == '__main__':
    _device = qroute.environment.device.GridComputerDevice(4, 4)
    _cirq = qroute.environment.circuits.circuit_generated_full_layer(len(_device), 3)
    _circuit = qroute.environment.circuits.CircuitRepDQN(_cirq)
    assert len(_circuit) == len(_device), "All qubits on target hardware need to be used once #FIXME"
    _agent = qroute.models.actor_critic.ActorCriticAgent(_device)
    train(_device, _circuit, _agent, training_episodes=500)
