import numpy as np
import tqdm
# import wandb

from qroute.metas import CombinerAgent, ReplayMemory, MemoryItem
from qroute.environment.env import step
from qroute.environment.circuits import CircuitRepDQN
from qroute.environment.device import DeviceTopology
from qroute.environment.state import CircuitStateDQN
from qroute.visualizers.solution_validator import validate_solution


def train_step(agent: CombinerAgent,
               device: DeviceTopology,
               circuit: CircuitRepDQN,
               memory: ReplayMemory,
               training_steps=500, episode_id=1):

    input_circuit = circuit
    state = CircuitStateDQN(input_circuit, device)
    solution_start, solution_moments = np.array(state.node_to_qubit), []

    state, total_reward, done, debugging_output = step(np.full(len(state.device.edges), False), state)
    solution_moments.append(debugging_output)
    if done:
        print("Episode %03d: The initial circuit is executable with no additional swaps" % episode_id)
        return
    progress_bar = tqdm.trange(training_steps)
    progress_bar.set_description('Episode %03d' % episode_id)

    for time in progress_bar:
        action, _ = agent.act(state)
        next_state, reward, done, debugging_output = step(action, state)
        total_reward += reward
        solution_moments.append(debugging_output)
        memory.store(MemoryItem(state=state, action=action, next_state=next_state, reward=reward, done=done))
        state = next_state

        if (time + 1) % 500 == 0:
            agent.replay(memory)

        progress_bar.set_postfix(total_reward=total_reward)
        if done:
            num_actions = time + 1
            result_circuit = validate_solution(input_circuit, solution_moments, solution_start, device)
            depth = len(result_circuit.moments)
            progress_bar.set_postfix(circuit_depth=depth, num_actions=num_actions, total_reward=total_reward)
            progress_bar.close()

            # print(solution_start, "\n", input_circuit.cirq, "\n", result_circuit, "\n", flush=True)
            # wandb.log({'Circuit Depth': depth,
            #            'Number of Actions': num_actions,
            #            'Input Circuit': str(input_circuit.cirq),
            #            'Output Circuit': str(result_circuit)})
            return solution_start, solution_moments, True

    agent.replay(memory)

    return solution_start, solution_moments, False
