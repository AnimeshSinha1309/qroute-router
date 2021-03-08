import copy
import sys

import numpy as np
import pygame

from qroute.environment.state import CircuitStateDQN
from qroute.environment.device import GridComputerDevice


class RenderedStateDQN:

    BLOCK_SIZE = 200
    LINE_COLOR = (0, 0, 0)
    NORMAL_COLOR = (230, 230, 230)
    CNOT_COLORS = [(255, 182, 193), (255, 207, 158)]
    SWAP_COLORS = [(144, 238, 144), (173, 216, 230)]

    def __init__(self, initial_state):
        self.state: CircuitStateDQN = initial_state
        # noinspection PyTypeChecker
        self.device: GridComputerDevice = self.state.device
        pygame.init()
        assert type(self.device) is GridComputerDevice, "Rendering is not supported for non-grid devices"
        self.screen = pygame.display.set_mode((
            self.device.cols * self.BLOCK_SIZE,
            self.device.rows * self.BLOCK_SIZE + self.BLOCK_SIZE // self.device.rows))
        pygame.display.set_caption("Routing Visualization")
        self.render()

    def render(self, cnots=None, swaps=None, time=1000):
        self.device: GridComputerDevice

        # Assigning colors to the rectangles
        colors = [self.NORMAL_COLOR for _ in range(len(self.device))]
        if cnots is not None:
            for idx, (n0, n1) in enumerate(cnots):
                colors[n0] = self.CNOT_COLORS[idx % len(self.CNOT_COLORS)]
                colors[n1] = self.CNOT_COLORS[idx % len(self.CNOT_COLORS)]
        if swaps is not None:
            for idx, (n0, n1) in enumerate(swaps):
                colors[n0] = self.SWAP_COLORS[idx % len(self.SWAP_COLORS)]
                colors[n1] = self.SWAP_COLORS[idx % len(self.SWAP_COLORS)]

        for x in range(self.device.cols):
            for y in range(self.device.rows):
                position = y * self.device.cols + x
                rect = pygame.Rect(
                    y * self.BLOCK_SIZE, x * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, colors[position], rect, 0, border_radius=0)

                text_value = str(self.state.node_to_qubit[position]) + '-' + \
                             str(self.state.node_to_qubit[self.state.target_nodes[position]]) \
                    if self.state.target_nodes[position] > -1 else str(self.state.node_to_qubit[position])
                font = pygame.font.SysFont("freesansbold", self.BLOCK_SIZE // 2, bold=False, italic=False)
                text = font.render(text_value, True, self.LINE_COLOR, colors[position])
                h, w = font.size(text_value)
                self.screen.blit(text, ((y + 0.5) * self.BLOCK_SIZE - h / 2, (x + 0.5) * self.BLOCK_SIZE - w / 2))

        for x in range(1, self.device.cols):
            pygame.draw.line(self.screen, self.LINE_COLOR,
                             (x * self.BLOCK_SIZE, 0), (x * self.BLOCK_SIZE, self.device.rows * self.BLOCK_SIZE), 2)
        for y in range(1, self.device.rows):
            pygame.draw.line(self.screen, self.LINE_COLOR,
                             (0, y * self.BLOCK_SIZE), (self.device.cols * self.BLOCK_SIZE, y * self.BLOCK_SIZE), 2)

        for x in range(1, self.device.cols * self.device.rows):
            pygame.draw.line(self.screen, self.NORMAL_COLOR,
                             (x * self.BLOCK_SIZE // self.device.rows, self.device.rows * self.BLOCK_SIZE),
                             (x * self.BLOCK_SIZE // self.device.rows, self.device.rows * self.BLOCK_SIZE +
                              self.BLOCK_SIZE // self.device.rows), 2)

        for idx, lock in enumerate(self.state.locked_edges):
            text_value = "1" if lock else "0"
            font = pygame.font.SysFont("freesansbold", self.BLOCK_SIZE // 2, bold=False, italic=False)
            text = font.render(text_value, True, self.NORMAL_COLOR, self.LINE_COLOR)
            h, w = font.size(text_value)
            self.screen.blit(text, ((idx + 0.5) * self.BLOCK_SIZE / self.device.rows - h / 2,
                                    (self.device.rows + 0.5 / self.device.rows) * self.BLOCK_SIZE - w / 2))

        pygame.display.update()
        pygame.time.wait(time)

    def update(self, new_state):
        self.state = new_state

    @staticmethod
    def respond(wait=True):
        print("Responding to Input, please press key!")
        while True:
            next_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit()
                elif (not wait) or (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
                    next_step = True
                    break
            if next_step:
                break

    def screenshot(self, idx):
        pygame.image.save(self.screen, f"paper/gameshots/shot_{idx}.jpg")


def simulate(action_chooser, input_state, state_renderer: RenderedStateDQN, forced_action=None):
    """
    Takes one step in the environment.
    This is actually the combination of 2 steps, the swaps in the current step and
    """
    output_state = copy.copy(input_state)
    # CNOTs
    cnots_executed = output_state.execute_cnot()
    cnot_action = np.array([gate in cnots_executed for gate in output_state.device.edges])
    output_state.update_locks(cnot_action, 1)
    # SWAPs
    seen_state = copy.copy(output_state)
    state_renderer.update(seen_state)  # The state as seen by the model, before CNOTs
    swap_action = action_chooser.act(output_state) if forced_action is None else forced_action
    swaps_executed = output_state.execute_swap(swap_action)
    state_renderer.render(cnots=cnots_executed, swaps=swaps_executed)  # Show the
    output_state.update_locks(swap_action, 1)
    # Return everything
    output_state.update_locks()
    return output_state


if __name__ == "__main__":
    from qroute.models.graph_dual import GraphDualModel
    from qroute.algorithms.deepmcts import MCTSAgent
    from qroute.memory.list import MemorySimple
    from qroute.environment.circuits import circuit_generated_randomly, CircuitRepDQN
    np.random.seed(42)

    device = GridComputerDevice(2, 2)
    model = GraphDualModel(device, True)
    memory = MemorySimple(0)
    agent = MCTSAgent(model, device, memory, search_depth=20)
    cirq = circuit_generated_randomly(num_qubits=4, num_cx=5)
    circuit = CircuitRepDQN(cirq, len(device))
    state = CircuitStateDQN(circuit, device)
    print(cirq)

    renderer = RenderedStateDQN(state)  # The original state before anything is touched

    for i in range(1, 1000 + 1):
        renderer.respond()
        state = simulate(agent, state, renderer)
        renderer.screenshot(i)
        if state.is_done():
            renderer.respond()
            break
