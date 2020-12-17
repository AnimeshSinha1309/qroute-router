from manim import *
import qroute


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

    for i in range(500):
        action, _ = agent.act(state)
        state, reward, done, next_gates_scheduled = qroute.environment.env.step(action, state)
        if done:
            output_circuit = qroute.visualizers.solution_validator.validate_solution(
                circuit, state.solution, initial_solution, device)
            print('Output Circuit')
            print(output_circuit)
            output_actions = qroute.visualizers.solution_validator.segment_ops_to_moments(
                state.solution, initial_solution)
            print(output_actions)
            return initial_solution, output_actions, circuit.circuit

    raise RuntimeError('Simulation could not find a solution')


class GridComputeScene(Scene):

    def __init__(self):
        super().__init__()
        self.GRID_SIZE = 2
        self.loc, self.ops, self.circuit = model_run(self.GRID_SIZE)
        print(self.loc)
        print(self.circuit)
        print(self.ops)
        self.done = np.zeros(self.GRID_SIZE ** 2, dtype=np.int)
        self.nodes, self.labels = [], []

    def construct(self):
        shifts = (np.arange(self.GRID_SIZE) - (self.GRID_SIZE - 1) / 2) * 4 / 3
        animations = []
        for i, x in enumerate(shifts):
            for j, y in enumerate(shifts):
                pos = self.GRID_SIZE * i + j
                square = Square(color=PINK).scale(0.8)
                separator = Text('-')
                qubit = DecimalNumber(number=self.loc[pos] + 1, num_decimal_places=0)\
                    .next_to(separator, LEFT, 0.1)
                target = DecimalNumber(number=0, num_decimal_places=0).next_to(separator, RIGHT, 0.1)
                node = VGroup(square, separator, qubit, target)
                node.shift(RIGHT * x).shift(DOWN * y).scale(0.7)
                self.nodes.append(node)
                self.labels.append(target)
                animations.append(ShowCreation(node))
        for pos in range(self.GRID_SIZE ** 2):
            self.target_update(pos)

        self.play(*animations)
        self.wait(3)
        for m in self.ops:
            self.moment(m)

    def target_update(self, pos):
        if self.labels[self.loc[pos]] is None:
            return []
        elif self.done[self.loc[pos]] >= len(self.circuit[self.loc[pos]]):
            outro = FadeOut(self.labels[self.loc[pos]])
            self.labels[self.loc[pos]] = None
            return [outro]
        else:
            self.labels[self.loc[pos]].set_value(
                self.circuit[self.loc[pos]][self.done[self.loc[pos]]] + 1)
            return []

    def moment(self, ops):
        print(ops)
        animations, outro = [], []
        for x, y, t in ops:
            if t == 'swap':
                a, o = self.swap(x, y)
            else:
                a, o = self.cnot(x, y)
            animations.extend(a)
            outro.extend(o)

        self.play(*animations)
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
