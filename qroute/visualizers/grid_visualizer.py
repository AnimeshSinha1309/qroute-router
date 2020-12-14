from manim import *


class GridComputeScene(Scene):

    def __init__(self):
        super().__init__()
        self.GRID_SIZE = 4
        self.ops = [[(2, 3, 'swap'), (7, 11, 'cnot'), (5, 6, 'swap')],
                    [(0, 1, 'swap'), (15, 11, 'cnot'), (8, 9, 'cnot')]]
        self.nodes, self.labels = [], []
        self.circuit_progress = np.zeros(self.GRID_SIZE ** 2, dtype=np.int)
        self.loc = np.arange(1, self.GRID_SIZE ** 2 + 1)
        self.shifts = np.linspace(-2, 2, self.GRID_SIZE)

    def construct(self):
        for i, x in enumerate(self.shifts):
            for j, y in enumerate(self.shifts):
                pos = i * 4 + j
                square = Square(color=PINK).scale(0.5).shift(RIGHT * x).shift(DOWN * y)
                text = Text(str(self.loc[pos]) + '-' + str(self.circuit_progress[pos])).scale(0.5)
                text.move_to(square.get_center())
                self.nodes.append(square)
                self.labels.append(text)
        animations = [ShowCreation(node) for node in self.nodes] + [ShowCreation(label) for label in self.labels]
        self.play(*animations)
        self.wait(3)
        for m in self.ops:
            self.moment(m)

    def moment(self, ops):
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
        x_i, x_j = x // 4, x % 4
        y_i, y_j = y // 4, y % 4

        rectangle = Rectangle(color=GREEN, width=(2.5 if x_i != y_i else 1.15), height=(2.5 if x_j != y_j else 1.15))\
            .move_to((self.nodes[x].get_center() + self.nodes[y].get_center()) / 2)
        animations = [ApplyMethod(self.nodes[x].move_to, self.nodes[y].get_center()),
                      ApplyMethod(self.nodes[y].move_to, self.nodes[x].get_center()),
                      ApplyMethod(self.labels[x].move_to, self.nodes[y].get_center()),
                      ApplyMethod(self.labels[y].move_to, self.nodes[x].get_center()),
                      FadeIn(rectangle)]
        outro = [FadeOut(rectangle)]
        self.nodes[x], self.nodes[y] = self.nodes[y], self.nodes[x]
        self.labels[x], self.labels[y] = self.labels[y], self.labels[x]
        self.loc[x], self.loc[y] = self.loc[y], self.loc[x]
        return animations, outro

    def cnot(self, x, y):
        x_i, x_j = x // 4, x % 4
        y_i, y_j = y // 4, y % 4

        rectangle = Rectangle(color=BLUE, width=(2.5 if x_i != y_i else 1.15), height=(2.5 if x_j != y_j else 1.15))\
            .move_to((self.nodes[x].get_center() + self.nodes[y].get_center()) / 2)
        animations = [FadeOut(self.labels[x]),
                      FadeOut(self.labels[y]),
                      FadeIn(rectangle)]
        outro = [FadeOut(rectangle)]

        self.circuit_progress[self.loc[x] - 1] += 1
        self.circuit_progress[self.loc[y] - 1] += 1
        self.labels[x] = Text(str(self.loc[x]) + '-' + str(self.circuit_progress[self.loc[x] - 1])).scale(0.5)
        self.labels[y] = Text(str(self.loc[y]) + '-' + str(self.circuit_progress[self.loc[y] - 1])).scale(0.5)
        self.labels[x].move_to(self.nodes[x].get_center())
        self.labels[y].move_to(self.nodes[y].get_center())
        animations.extend([FadeIn(self.labels[x]), FadeIn(self.labels[y])])
        return animations, outro
