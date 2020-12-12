from manim import *


class GridComputeScene(Scene):

    def __init__(self):
        super().__init__()
        self.GRID_SIZE = 4
        self.ops = [[(2, 3, 'swap'), (7, 11, 'cnot'), (5, 6, 'swap')],
                    [(0, 1, 'swap'), (15, 11, 'cnot'), (8, 9, 'cnot')]]
        self.nodes, self.labels = [], []
        self.circuit_progress = np.zeros(self.GRID_SIZE ** 2, dtype=np.int)
        self.initial_location = np.arange(1, self.GRID_SIZE ** 2 + 1)
        self.shifts = np.linspace(-2, 2, self.GRID_SIZE)

    def construct(self):
        for i, x in enumerate(self.shifts):
            for j, y in enumerate(self.shifts):
                square = Square(color=PINK).scale(0.5)
                pos = i * 4 + j
                text = Text(str(self.initial_location[pos]) + '-' + str(self.circuit_progress[pos])).scale(0.5)
                node = VGroup(square, text)
                node.shift(RIGHT * x)
                node.shift(DOWN * y)
                self.nodes.append(node)
                self.labels.append(text)
        animations = [ShowCreation(node) for node in self.nodes]
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
        animations = []
        rectangle = None

        if x_j != y_j:
            animations.append(ApplyMethod(self.nodes[x].shift, DOWN * (self.shifts[y_j] - self.shifts[x_j])))
            rectangle = Rectangle(color=GREEN, width=1.15, height=2.5)
            rectangle.shift(RIGHT * self.shifts[x_i])
            rectangle.shift(DOWN * (self.shifts[x_j] + self.shifts[y_j]) / 2)
        if x_i != y_i:
            animations.append(ApplyMethod(self.nodes[x].shift, RIGHT * (self.shifts[y_i] - self.shifts[x_i])))
            rectangle = Rectangle(color=GREEN, width=2.5, height=1.15)
            rectangle.shift(DOWN * self.shifts[x_j])
            rectangle.shift(RIGHT * (self.shifts[x_i] + self.shifts[y_i]) / 2)
        if x_j != y_j:
            animations.append(ApplyMethod(self.nodes[y].shift, UP * (self.shifts[y_j] - self.shifts[x_j])))
            rectangle = Rectangle(color=GREEN, width=1.15, height=2.5)
            rectangle.shift(RIGHT * self.shifts[x_i])
            rectangle.shift(DOWN * (self.shifts[x_j] + self.shifts[y_j]) / 2)
        if x_i != y_i:
            animations.append(ApplyMethod(self.nodes[y].shift, LEFT * (self.shifts[y_i] - self.shifts[x_i])))
            rectangle = Rectangle(color=GREEN, width=2.5, height=1.15)
            rectangle.shift(DOWN * self.shifts[x_j])
            rectangle.shift(RIGHT * (self.shifts[x_i] + self.shifts[y_i]) / 2)

        animations.append(FadeIn(rectangle))
        outro = [FadeOut(rectangle)]
        self.nodes[x], self.nodes[y] = self.nodes[y], self.nodes[x]
        return animations, outro

    def cnot(self, x, y):
        x_i, x_j = x // 4, x % 4
        y_i, y_j = y // 4, y % 4
        animations = []
        rectangle = None

        if x_j != y_j:
            rectangle = Rectangle(color=BLUE, width=1.15, height=2.5)
            rectangle.shift(RIGHT * self.shifts[x_i])
            rectangle.shift(DOWN * (self.shifts[x_j] + self.shifts[y_j]) / 2)
        if x_i != y_i:
            rectangle = Rectangle(color=BLUE, width=2.5, height=1.15)
            rectangle.shift(DOWN * self.shifts[x_j])
            rectangle.shift(RIGHT * (self.shifts[x_i] + self.shifts[y_i]) / 2)
        if x_j != y_j:
            rectangle = Rectangle(color=BLUE, width=1.15, height=2.5)
            rectangle.shift(RIGHT * self.shifts[x_i])
            rectangle.shift(DOWN * (self.shifts[x_j] + self.shifts[y_j]) / 2)
        if x_i != y_i:
            rectangle = Rectangle(color=BLUE, width=2.5, height=1.15)
            rectangle.shift(DOWN * self.shifts[x_j])
            rectangle.shift(RIGHT * (self.shifts[x_i] + self.shifts[y_i]) / 2)

        self.circuit_progress[x] += 1
        self.circuit_progress[y] += 1

        animations.append(FadeIn(rectangle))
        outro = [FadeOut(rectangle)]
        return animations, outro
