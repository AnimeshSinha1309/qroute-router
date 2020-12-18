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
    reward = 0

    # Execute the gates you have on queue
    gates_executed = state.execute_next()
    reward += gates_executed * qroute.hyperparams.REWARD_GATE

    # Swaps the required qubits and collects rewards for the gain in distances
    pre_swap_distances = state.target_distance
    state.execute_swap(action)
    post_swap_distances = state.target_distance
    reward += qroute.hyperparams.REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(0, 100, pre_swap_distances - post_swap_distances))

    # Return everything
    return state, reward, state.is_done(), None
