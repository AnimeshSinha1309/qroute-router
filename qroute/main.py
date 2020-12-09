import copy
import collections
import logging

import numpy as np
import tqdm

import qroute

logging.basicConfig(level=logging.DEBUG)


def train(device: qroute.environment.device.DeviceTopology,
          circuit: qroute.environment.circuits.CircuitRepDQN,
          agent: qroute.models.double_dqn.DoubleDQNAgent,
          training_episodes=350, training_steps=500):

    num_actions_deque = collections.deque(maxlen=50)
    time_between_model_updates = 5

    memory = qroute.environment.memory.MemoryPER(500)

    # Fill up memory tree
    while memory.tree.used_up_capacity < memory.tree.capacity:
        state = qroute.environment.state.CircuitStateDQN(circuit, device)
        state.generate_starting_state()

        for time in range(training_steps):
            action, _ = agent.act(state)
            next_state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
            memory.store((state, reward, next_state, done))
            state = next_state

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                break

    # Training the agent
    for e in range(training_episodes):
        state = qroute.environment.state.CircuitStateDQN(circuit, device)
        state.generate_starting_state()

        logging.debug("Starting Training on Episode %d" % e)

        for time in tqdm.trange(training_steps):
            temp_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            action, _ = agent.act(state)
            new_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            assert temp_state == new_state, "State not preserved when selecting action"

            next_state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
            memory.store((state, reward, next_state, done))
            state = next_state

            if done:
                num_actions = time+1
                num_actions_deque.append(num_actions)
                avg_time = np.mean(num_actions_deque)
                logging.debug("Number of actions: {}, average: {:.5}".format(num_actions, avg_time))
                logging.debug("Final positions in episodes: %s" % str(next_state.qubit_locations))

            agent.replay(memory)

            if time % time_between_model_updates == 0:
                agent.update_target_model()


if __name__ == '__main__':
    _device = qroute.environment.device.GridComputerDevice(4, 4)
    _circuit = qroute.environment.circuits.CircuitRepDQN(
        qroute.environment.circuits.circuit_generated_full_layer(len(_device)))
    _agent = qroute.models.double_dqn.DoubleDQNAgent(_device)
    train(_device, _circuit, _agent)
