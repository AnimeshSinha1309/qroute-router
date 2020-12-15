import copy
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

    memory = qroute.environment.memory.MemoryPER(500)

    # Fill up memory tree
    while memory.tree.used_up_capacity < memory.tree.capacity:
        state = qroute.environment.state.CircuitStateDQN(circuit, device)
        state.generate_starting_state()

        progress_bar = tqdm.trange(training_steps)
        progress_bar.set_description('Initial Setup')
        for time in progress_bar:
            action, _ = agent.act(state)
            next_state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
            memory.store((state, reward, next_state, done))
            state = next_state

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                break
        progress_bar.close()

    # Training the agent
    for e in range(training_episodes):
        state = qroute.environment.state.CircuitStateDQN(circuit, device)
        state.generate_starting_state()
        starting_locations = np.array(state.qubit_locations)

        progress_bar = tqdm.trange(training_steps)
        progress_bar.set_description('Episode %03d' % (e + 1))
        for time in progress_bar:
            temp_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            action, _ = agent.act(state)
            new_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            assert temp_state == new_state, "State not preserved when selecting action"

            next_state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
            memory.store((state, reward, next_state, done))
            state = next_state

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                avg_time = np.mean(num_actions_deque)
                depth = qroute.visualizers.solution_validator.validate_solution(
                    circuit, state.solution, starting_locations, device)
                progress_bar.set_postfix(circuit_depth=depth, num_actions=num_actions, avg_actions=avg_time)
                # wandb.log({'Circuit Depth': depth,
                #            'Number of Current Actions': num_actions,
                #            'Number of Average Actions': num_actions})
                progress_bar.close()
                break

            agent.replay(memory)


if __name__ == '__main__':
    _device = qroute.environment.device.GridComputerDevice(4, 4)
    _cirq = qroute.environment.circuits.circuit_generated_full_layer(len(_device), 3)
    _circuit = qroute.environment.circuits.CircuitRepDQN(_cirq)
    assert len(_circuit) == len(_device), "All qubits on target hardware need to be used once #FIXME"
    _agent = qroute.models.actor_critic.ActorCriticAgent(_device)
    train(_device, _circuit, _agent, training_episodes=500)
