import copy
import collections

import numpy as np

from qroute.environment.state import CircuitStateDQN
import qroute.hyperparams


Moment = collections.namedtuple('Moment', ['cnots', 'swaps', 'reward'])


def step(action, input_state: CircuitStateDQN):
    """
    Takes one step in the environment
    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: state, the state in the upcoming step
    :return: reward, the reward obtained from the operations in the current step
    :return: done, True if execution is complete, False otherwise
    :return: debugging output, Moment containing the gates executed and the reward obtained
    """
    state: CircuitStateDQN = copy.copy(input_state)
    # Execute the gates you have on queue
    cnots_executed = state.execute_cnot()
    gate_reward = len(cnots_executed) * qroute.hyperparams.REWARD_GATE
    # Swaps the required qubits and collects rewards for the gain in distances
    pre_swap_distances = np.copy(state.target_distance)
    swaps_executed = state.execute_swap(action)
    post_swap_distances = np.copy(state.target_distance)
    swap_reward_dec = qroute.hyperparams.REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, 0, 1000))
    swap_reward_inc = qroute.hyperparams.REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, -1000, 0))
    # Check if the circuit is done executing
    done = state.is_done()
    reward_completion = qroute.hyperparams.REWARD_CIRCUIT_COMPLETION if done else 0
    # Return everything
    reward = gate_reward + swap_reward_inc + swap_reward_dec + reward_completion
    debugging_output = Moment(cnots_executed, swaps_executed, reward)
    return state, reward, done, debugging_output
