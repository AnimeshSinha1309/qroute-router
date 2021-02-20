import copy
import collections

import numpy as np

from ..environment.state import CircuitStateDQN
from ..hyperparams import *


Moment = collections.namedtuple('Moment', ['cnots', 'swaps', 'reward'])


def step(action, input_state: CircuitStateDQN):
    """
    Takes one step in the environment.
    This is actually the combination of 2 steps, the swaps in the current step and
    the cnot and setup for the next step.
    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: state, the state in the upcoming step
    :return: reward, the reward obtained from the operations in the current step
    :return: done, True if execution is complete, False otherwise
    :return: debugging output, Moment containing the gates executed and the reward obtained
    """
    state: CircuitStateDQN = copy.copy(input_state)
    assert not np.any(np.bitwise_and(state.locked_edges, action)), "Bad Action"
    # Swaps the required qubits and collects rewards for the gain in distances
    pre_swap_distances = np.copy(state.target_distance)
    swaps_executed = state.execute_swap(action)
    post_swap_distances = np.copy(state.target_distance)
    swap_reward_dec = REWARD_DISTANCE_REDUCTION * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, 0, 1000))
    swap_reward_inc = PENALTY_DISTANCE_INCREASE * np.sum(
        np.clip(pre_swap_distances - post_swap_distances, -1000, 0))
    state.update_locks(action, 1)
    state.update_locks()
    # Execute the gates you have on queue
    cnots_executed = state.execute_cnot()
    gate_reward = len(cnots_executed) * REWARD_GATE
    cnot_action = np.array([gate in cnots_executed for gate in state.device.edges])
    state.update_locks(cnot_action, 1)
    # Check if the circuit is done executing
    done = state.is_done()
    reward_completion = REWARD_CIRCUIT_COMPLETION if done else 0
    # Return everything
    reward = gate_reward + swap_reward_inc + swap_reward_dec + reward_completion + REWARD_TIMESTEP
    debugging_output = Moment(cnots_executed, swaps_executed, reward)
    return state, reward, done, debugging_output


def evaluate(action, input_state: CircuitStateDQN):
    """
    Takes one step in the environment
    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: state, the state in the upcoming step
    :return: reward, the reward obtained from the operations in the current step
    :return: done, True if execution is complete, False otherwise
    :return: debugging output, Moment containing the gates executed and the reward obtained
    """
    assert not np.any(np.bitwise_and(input_state.locked_edges, action)), "Bad Action"
    _next_state, reward, _done, _debug = step(action, input_state)
    return reward
