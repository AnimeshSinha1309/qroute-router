import numpy as np
import tqdm
# import wandb

import qroute


def train_step(agent: qroute.metas.CombinerAgent,
               device: qroute.environment.device.DeviceTopology,
               circuit: qroute.environment.circuits.CircuitRepDQN,
               memory: qroute.metas.ReplayMemory,
               training_steps=500, episode_id=1):

    input_circuit = circuit
    state = qroute.environment.state.CircuitStateDQN(input_circuit, device)
    solution_start, solution_moments = np.array(state.node_to_qubit), []

    progress_bar = tqdm.trange(training_steps)
    progress_bar.set_description('Episode %03d' % episode_id)
    total_reward = 0

    for time in progress_bar:
        action, _ = agent.act(state)
        next_state, reward, done, debugging_output = qroute.environment.env.step(action, state)
        total_reward += reward
        solution_moments.append(debugging_output)
        memory.store(qroute.metas.MemoryItem(state=state, action=action, next_state=next_state,
                                             reward=reward, done=done))
        state = next_state

        if (time + 1) % 500 == 0:
            agent.replay(memory)

        progress_bar.set_postfix(total_reward=total_reward)
        if done:
            num_actions = time + 1
            result_circuit = qroute.visualizers.solution_validator.validate_solution(
                input_circuit, solution_moments, solution_start, device)
            depth = len(result_circuit.moments)
            progress_bar.set_postfix(circuit_depth=depth, num_actions=num_actions, total_reward=total_reward)
            progress_bar.close()
            # wandb.log({'Circuit Depth': depth,
            #            'Number of Actions': num_actions,
            #            'Input Circuit': str(input_circuit.cirq),
            #            'Output Circuit': str(result_circuit)})
            return solution_start, solution_moments, True

    agent.replay(memory)

    return solution_start, solution_moments, False
