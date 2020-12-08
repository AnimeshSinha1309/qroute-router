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
    state.execute_next()  # Can also collect the rewards here, but not doing that

    # Swaps the required qubits and collects rewards for the gain in distances

    pre_swap_distances = state.calculate_distances_summary(state.qubit_locations, state.qubit_targets)

    swap_edge_indices = np.where(np.array(action) == 1)[0]
    swap_edges = [state.device.edges[i] for i in swap_edge_indices]

    for (node1, node2) in swap_edges:
        # FIXME: This is weird, why is qubit_locations indexed with nodes?
        state.qubit_locations[node1], state.qubit_locations[node2] = \
            state.qubit_locations[node2], state.qubit_locations[node1]
    post_swap_distances = state.calculate_distances_summary(state.qubit_locations, state.qubit_targets)
    distance_reduction_reward = 0

    for q in range(len(state.circuit)):
        if post_swap_distances[q] < pre_swap_distances[q]:
            distance_reduction_reward += qroute.hyperparams.REWARD_DISTANCE_REDUCTION
    gates_scheduled = state.next_gates()
    post_swap_reward = len(gates_scheduled) * qroute.hyperparams.REWARD_GATE

    reward = post_swap_reward + distance_reduction_reward

    next_state = copy.copy(state)
    return next_state, reward, next_state.is_done(), gates_scheduled
