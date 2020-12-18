import collections
# noinspection PyPackageRequirements
from manim import *

import qroute


Moment = collections.namedtuple('Moment', ['ops', 'reward'])


def model_run(grid_size):
    device = qroute.environment.device.GridComputerDevice(grid_size, grid_size)
    cirq = qroute.environment.circuits.circuit_generated_full_layer(len(device), 1)
    circuit = qroute.environment.circuits.CircuitRepDQN(cirq)
    agent = qroute.models.actor_critic.ActorCriticAgent(device)

    print('Input Circuit')
    print(circuit.cirq)

    state = qroute.environment.state.CircuitStateDQN(circuit, device)
    state.generate_starting_state()
    initial_solution = state.qubit_locations

    output_moments = []
    ops_read_in = 0

    for i in range(500):
        action, _ = agent.act(state)
        state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
        output_moments.append(Moment(ops=state.solution[ops_read_in:], reward=reward))
        ops_read_in = len(state.solution)
        print(state.solution, state.circuit_progress)
        if done:
            output_circuit = qroute.visualizers.solution_validator.validate_solution(
                circuit, state.solution, initial_solution, device)
            print('Output Circuit')
            print(output_circuit)
            return initial_solution, output_moments, circuit.circuit

    raise RuntimeError('Simulation could not find a solution')


print(model_run(2))


class GridComputeScene(Scene):

    def __init__(self):
        super().__init__()
        self.GRID_SIZE = 2
        self.loc, self.moments, self.circuit = model_run(self.GRID_SIZE)
        self.done = np.zeros(self.GRID_SIZE ** 2, dtype=np.int)
        self.nodes, self.labels = [], []
        self.reward_label = None

    def construct(self):
        shifts = (np.arange(self.GRID_SIZE) - (self.GRID_SIZE - 1) / 2) * 4 / 3
        animations = []
        for i, x in enumerate(shifts):
            for j, y in enumerate(shifts):
                pos = self.GRID_SIZE * i + j
                square = Square(color=PINK).scale(0.8)
                separator = Text('-')
                qubit = DecimalNumber(number=self.loc[pos], num_decimal_places=0)\
                    .next_to(separator, LEFT, 0.1)
                target = DecimalNumber(number=0, num_decimal_places=0).next_to(separator, RIGHT, 0.1)
                node = VGroup(square, separator, qubit, target)
                node.shift(RIGHT * x).shift(DOWN * y).scale(0.7)
                self.nodes.append(node)
                self.labels.append(target)
                animations.append(ShowCreation(node))
        self.play(*animations)
        for pos in range(self.GRID_SIZE ** 2):
            self.target_update(pos)
        self.reward_label = DecimalNumber(number=0, num_decimal_places=0).next_to(
            self.nodes[self.GRID_SIZE ** 2 - self.GRID_SIZE + 0], RIGHT, 1)
        self.play(ShowCreation(self.reward_label))
        self.wait(3)
        for m in self.moments:
            self.moment(m)

    def target_update(self, pos):
        if self.labels[pos] is None:
            return []
        elif self.done[self.loc[pos]] >= len(self.circuit[self.loc[pos]]):
            square = Square(color=PINK).scale(0.8)
            qubit = DecimalNumber(number=self.loc[pos], num_decimal_places=0)
            node = VGroup(square, qubit).move_to(self.nodes[pos].get_center()).scale(0.7)
            outro = [FadeOut(self.nodes[pos]), FadeIn(node)]
            self.nodes[pos] = node
            self.labels[pos] = None
            return outro
        else:
            self.labels[pos].set_value(
                self.circuit[self.loc[pos]][self.done[self.loc[pos]]])
            return []

    def moment(self, moment: Moment):
        print('Moment:', moment.ops)
        animations, outro = [], []
        for x, y, t in moment.ops:
            if t == 'swap':
                a, o = self.swap(x, y)
            else:
                a, o = self.cnot(x, y)
            animations.extend(a)
            outro.extend(o)

        self.play(*animations)
        self.reward_label.set_value(moment.reward)
        self.wait(2)
        self.play(*outro)
        self.wait(1)

    def swap(self, x, y):
        x_i, x_j = x // self.GRID_SIZE, x % self.GRID_SIZE
        y_i, y_j = y // self.GRID_SIZE, y % self.GRID_SIZE
        rectangle = Rectangle(
            color=GREEN, width=(2.5 if x_i != y_i else 1.15), height=(2.5 if x_j != y_j else 1.15))\
            .move_to((self.nodes[x].get_center() + self.nodes[y].get_center()) / 2)
        animations = [ApplyMethod(self.nodes[x].move_to, self.nodes[y].get_center()),
                      ApplyMethod(self.nodes[y].move_to, self.nodes[x].get_center()),
                      FadeIn(rectangle)]
        outro = [FadeOut(rectangle)]
        self.nodes[x], self.nodes[y] = self.nodes[y], self.nodes[x]
        self.loc[x], self.loc[y] = self.loc[y], self.loc[x]
        self.labels[x], self.labels[y] = self.labels[y], self.labels[x]
        return animations, outro

    def cnot(self, x, y):
        x_i, x_j = x // self.GRID_SIZE, x % self.GRID_SIZE
        y_i, y_j = y // self.GRID_SIZE, y % self.GRID_SIZE
        rectangle = Rectangle(
            color=BLUE, width=(2.5 if x_i != y_i else 1.15), height=(2.5 if x_j != y_j else 1.15))\
            .move_to((self.nodes[x].get_center() + self.nodes[y].get_center()) / 2)
        animations = [FadeIn(rectangle)]
        outro = [FadeOut(rectangle)]

        self.done[self.loc[x]] += 1
        self.done[self.loc[y]] += 1
        outro.extend(self.target_update(x))
        outro.extend(self.target_update(y))
        return animations, outro
