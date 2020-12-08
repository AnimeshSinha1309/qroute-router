import copy
import collections

import numpy as np
import tqdm

import qroute


def train(device: qroute.environment.device.DeviceTopology,
          circuit: qroute.environment.circuits.CircuitRepDQN,
          agent: qroute.models.double_dqn.DoubleDQNAgent,
          training_episodes=350, training_steps=500):

    num_actions_deque = collections.deque(maxlen=50)
    batch_size = 32
    time_between_model_updates = 5

    # --- Fill up memory tree ---
    while agent.used_up_memory_capacity() < agent.memory_size:
        state, gates_scheduled = reset_environment_state(environment, circuit_generation_function)

        for time in range(500):
            action, _ = agent.act(state)
            next_state, reward, done, next_gates_scheduled = environment.step(action, state)
            agent.remember(state, reward, next_state, done)
            state = next_state

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                break

    # --- Training ---
    for e in range(training_episodes):
        state, gates_scheduled = reset_environment_state(environment, circuit_generation_function)

        if should_print:
            print("Episode", e, "starting positions\n",
                  np.reshape(state.qubit_locations, (environment.rows, environment.cols)))

        for time in tqdm.trange(training_steps):
            temp_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            action, _ = agent.act(state)
            new_state: qroute.environment.state.CircuitStateDQN = copy.copy(state)
            assert temp_state == new_state, "State not preserved when selecting action"

            next_state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
            agent.remember(state, reward, next_state, done)
            state = next_state

            if done:
                num_actions = time+1
                num_actions_deque.append(num_actions)
                avg_time = np.mean(num_actions_deque)

                if should_print:
                    print("Number of actions: {}, average: {:.5}".format(num_actions, avg_time))
                    print("Final positions\n", np.reshape(next_state.qubit_locations[0:device],
                                                          (environment.rows, environment.cols)), '\n')
                break
            agent.replay(batch_size)

            if time % time_between_model_updates == 0:
                agent.update_target_model()


def run():
    for i in range(1):
        device = qroute.environment.device.GridComputerDevice(4, 4)
        circuit = qroute.environment.circuits.CircuitRepDQN(qroute.environment.circuits.circuit_generated_full_layer(5))
        agent = qroute.models.double_dqn.DoubleDQNAgent(device)
        train(device, circuit, agent)


if __name__ == '__main__':
    run()
