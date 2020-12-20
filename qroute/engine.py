import numpy as np
import tqdm
# import wandb

import qroute


def train_step(agent: qroute.algorithms.actanneal.AnnealerAct,
               device: qroute.environment.device.DeviceTopology,
               circuit: qroute.environment.circuits.CircuitRepDQN,
               memory: qroute.memory.list.MemorySimple,
               training_steps=500, episode_id=1):

    input_circuit = circuit
    state = qroute.environment.state.CircuitStateDQN(input_circuit, device)
    solution_start, solution_moments = np.array(state.node_to_qubit), []

    progress_bar = tqdm.trange(training_steps)
    progress_bar.set_description('Episode %03d' % episode_id)
    for time in progress_bar:
        action, _ = agent.act(state)
        next_state, reward, done, debugging_output = qroute.environment.env.step(action, state)
        solution_moments.append(debugging_output)
        memory.store((state, reward, next_state, done))
        # noinspection PyUnusedLocal
        state = next_state

        if done:
            num_actions = time + 1
            result_circuit = qroute.visualizers.solution_validator.validate_solution(
                input_circuit, solution_moments, solution_start, device)
            depth = len(result_circuit.moments)
            progress_bar.set_postfix(circuit_depth=depth, num_actions=num_actions)
            progress_bar.close()

            # wandb.log({'Circuit Depth': depth,
            #            'Number of Actions': num_actions,
            #            'Input Circuit': str(input_circuit.cirq),
            #            'Output Circuit': str(result_circuit)})
            return solution_start, solution_moments, True

    agent.replay(memory)
    return solution_start, solution_moments, False
