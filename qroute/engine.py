import os

import numpy as np
import tqdm
import torch
# import wandb

from .metas import CombinerAgent, ReplayMemory
from .environment.env import step
from .environment.circuits import CircuitRepDQN, circuit_to_json
from .environment.device import DeviceTopology
from .environment.state import CircuitStateDQN
from .visualizers.solution_validator import validate_solution


def train_step(agent: CombinerAgent,
               device: DeviceTopology,
               circuit: CircuitRepDQN,
               training_steps=100000, episode_id="Unnamed Run"):

    os.makedirs("./test/test_results", exist_ok=True)
    input_circuit = circuit
    state = CircuitStateDQN(input_circuit, device)
    solution_start, solution_moments = np.array(state.node_to_qubit), []

    state, total_reward, done, debugging_output = step(np.full(len(state.device.edges), False), state)
    solution_moments.append(debugging_output)
    if done:
        print("Episode %03d: The initial circuit is executable with no additional swaps" % episode_id)
        return
    progress_bar = tqdm.tqdm(total=len(list(circuit.cirq.all_operations())))
    progress_bar.set_description(episode_id)

    for time in range(2, training_steps + 1):
        action = agent.act(state)
        assert not np.any(np.bitwise_and(state.locked_edges, action)), "Bad Action"

        next_state, reward, done, debugging_output = step(action, state)
        total_reward += reward
        solution_moments.append(debugging_output)
        progress_bar.update(len(debugging_output.cnots))
        state = next_state

        if (time + 1) % 1000 == 0:
            agent.replay()
            torch.save(agent.model.state_dict(), "model-weights.h5")

        progress_bar.set_postfix(total_reward=total_reward, time=time)
        if done:
            result_circuit = validate_solution(input_circuit, solution_moments, solution_start, device)
            circuit_to_json(result_circuit, ("./test/test_results/%s.json" % episode_id))
            depth = len(result_circuit.moments)
            progress_bar.set_postfix(circuit_depth=depth, total_reward=total_reward, time=time)
            progress_bar.close()
            # print(solution_start, "\n", input_circuit.cirq, "\n", result_circuit, "\n", flush=True)
            # wandb.log({'Circuit Depth': depth,
            #            'Number of Actions': num_actions,
            #            'Input Circuit': str(input_circuit.cirq),
            #            'Output Circuit': str(result_circuit)})
            agent.replay()
            return solution_start, solution_moments, True

    agent.replay()
    torch.save(agent.model.state_dict(), "model-weights.h5")

    return solution_start, solution_moments, False
