import copy
import numpy as np


from qroute.environment.state import CircuitStateDQN
import qroute.hyperparams


def step(action, input_state: CircuitStateDQN):
    """
    Takes one step in the environment

    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: State, the state in the upcoming step
    """
    state: CircuitStateDQN = copy.copy(input_state)
    # Execute the gates you have on queue
    gates_executed = state.execute_next()
    gate_reward = len(gates_executed) * qroute.hyperparams.REWARD_GATE
    # Swaps the required qubits and collects rewards for the gain in distances
    pre_swap_distances = state.target_distance
    state.execute_swap(action)
    post_swap_distances = state.target_distance
    swap_reward_dec = qroute.hyperparams.REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, 0, 1000))
    swap_reward_inc = qroute.hyperparams.REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, -1000, 0))
    # Check if the circuit is done executing
    done = state.is_done()
    reward_completion = qroute.hyperparams.REWARD_CIRCUIT_COMPLETION if done else 0
    # Return everything
    reward = gate_reward + swap_reward_inc + swap_reward_dec + reward_completion
    return state, reward, done, gates_executed
