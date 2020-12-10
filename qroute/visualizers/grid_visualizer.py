import manimlib.imports as manim
import numpy as np


class MatrixBase:
    v_buff = 0.8
    h_buff = 1.3

    def __init__(self, matrix: np.ndarray, walker_idx: tuple, text_color=manim.WHITE):
        self.matrix = matrix
        self.dimensions = matrix.shape

        value_fn = lambda x, y: self.dimensions[1] * x + y + 1
        mobject_fn = lambda x, y: manim.Integer(number=value_fn(x, y), color=text_color)
        self.grid = np.array([[mobject_fn(i, j) for j in range(self.dimensions[1])] for i in range(self.dimensions[0])])

        self.walker_idx = walker_idx
        self.original_value = value_fn(*walker_idx)

        self.position_elements(self.grid)

    def compute_offset(self, i, j):
        offset = 1
        return (i - offset) * self.v_buff * manim.DOWN + (j - offset) * self.h_buff * manim.RIGHT

    def position_element(self, elm, i, j, aligned=manim.ORIGIN):
        elm.move_to(self.compute_offset(i, j), aligned)

    def position_elements(self, grid, aligned=manim.ORIGIN):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                self.position_element(grid[i, j], i, j, aligned=aligned)

    @staticmethod
    def get_transformations(elms, permutation, cnt):
        target_coords = [elms[permutation[i]].get_center() for i in range(cnt)]
        transformations = [manim.ApplyMethod(elms[i].move_to, target_coords[i]) for i in range(cnt)]

        return transformations

    def shuffle_row(self, row_idx, permutation_matrix: list):
        elms = np.copy(self.grid[row_idx, :].ravel())
        self.grid[row_idx, permutation_matrix] = elms

        cnt = self.dimensions[1]
        return self.get_transformations(elms, permutation_matrix, cnt)

    def shuffle_col(self, col_idx, permutation_matrix):
        elms = np.copy(self.grid[:, col_idx].ravel())
        self.grid[permutation_matrix, col_idx] = elms

        cnt = self.dimensions[0]
        return self.get_transformations(elms, permutation_matrix, cnt)

    def get_render_objects(self):
        return [self.grid.ravel()]


class SimpleMatrix(MatrixBase):
    pass


class BorderMatrix(MatrixBase):
    ORG_COLOR = manim.GREEN_E
    CURR_COLOR = manim.BLUE_E

    def __init__(self, matrix: np.ndarray, walker_idx: tuple, color=manim.WHITE):
        super().__init__(matrix, walker_idx, color)
        self.border = None
        self.add_borders()

    def add_borders(self):
        walker: manim.Mobject = self.grid[self.walker_idx]
        x, y = self.walker_idx

        self.border = manim.Rectangle(stroke_color=self.ORG_COLOR, width=self.h_buff, height=self.v_buff)
        self.position_element(self.border, x, y)

        curr_marker = manim.Rectangle(stroke_color=self.CURR_COLOR, width=self.h_buff, height=self.v_buff)
        self.position_element(curr_marker, x, y)
        walker.add_to_back(curr_marker)

    def get_render_objects(self):
        return super().get_render_objects() + [[self.border]]


COLORS = [manim.GREEN_A, manim.BLUE_A, manim.RED_A]
BORDER_COLORS = [manim.GREEN_E, manim.BLUE_E, manim.RED_E]


class BestMatrix(BorderMatrix):

    def __init__(self, matrix: np.ndarray, walker_idx: tuple):
        super().__init__(matrix, walker_idx, color=manim.BLACK)
        self.tiles = None
        self.add_backgroundtiles()

    def add_backgroundtiles(self):
        self.tiles = np.empty(self.dimensions, dtype=manim.Mobject)

        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                idx = None

                if i == self.walker_idx[0] and j == self.walker_idx[1]:
                    idx = 0
                elif i == self.walker_idx[0] or j == self.walker_idx[1]:
                    idx = 1
                else:
                    idx = 2

                border_color, color = BORDER_COLORS[idx], COLORS[idx]

                tile = manim.Rectangle(fill_color=color, fill_opacity=1, stroke_color=border_color,
                                       color=color, width=self.h_buff, height=self.v_buff)
                self.tiles[i, j] = tile

        self.position_elements(self.tiles)

    def get_render_objects(self):
        return [self.tiles.ravel()] + super().get_render_objects()


class MarkovChainThreeState:
    def __init__(self):
        self.circles = []
        self.directions = [3 * manim.LEFT, manim.ORIGIN, 3 * manim.RIGHT]

        for i in range(3):
            circ = manim.Circle(fill_color=COLORS[i], fill_opacity=1, stroke_color=BORDER_COLORS[i],
                                color=COLORS[i], radius=0.5)
            circ.move_to(self.directions[i])
            self.circles.append(circ)

        ltr_offset = np.array((0.8, 0., 0.))
        up = np.array((0., 0.2, 0))
        up_offset = [ltr_offset + up, -ltr_offset + up]
        down_offset = [ltr_offset - up, -ltr_offset - up]
        self_loop_offsets = [0.2 * manim.LEFT + manim.UP, 0.2 * manim.RIGHT + manim.UP]
        self.arrows = []

        def add_between(x, y):
            assert (x < y)
            self.arrows.append(manim.CurvedArrow(self.directions[y] + up_offset[1], self.directions[x] + up_offset[0]))
            self.arrows.append(manim.CurvedArrow(self.directions[x] + down_offset[0], self.directions[y] + down_offset[1]))

        def self_loop(x):
            self.arrows.append(
                manim.CurvedArrow(self.directions[x] + self_loop_offsets[0],
                                  self.directions[x] + self_loop_offsets[1],
                                  num_components=20))

        add_between(0, 1)
        add_between(1, 2)

        self_loop(1)
        self_loop(2)

    def get_render_objects(self):
        return self.circles + self.arrows


class MatrixTestScene(manim.Scene):
    def set_ticker(self):
        self.ticker = manim.TexMobject(f"{str(self.tick_value)} / 10")
        self.ticker.move_to(2 * manim.UP)

    def update_ticker(self):
        self.remove(self.ticker)
        self.tick_value += 1
        self.set_ticker()
        self.add(self.ticker)

    def play_with(self, mat):
        added_objects = mat.get_render_objects()
        for x in added_objects:
            self.add(*x)
        self.add(self.ticker)

        run_time = 1.5
        self.play(*mat.shuffle_row(2, [1, 2, 0, 3]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_col(2, [3, 1, 0, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_row(0, [0, 3, 1, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_row(0, [1, 0, 3, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_col(0, [3, 0, 2, 1]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_row(3, [3, 0, 1, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_col(3, [1, 0, 3, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_col(3, [3, 0, 1, 2]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_row(1, [0, 3, 2, 1]), run_time=run_time)
        self.update_ticker()
        self.play(*mat.shuffle_col(1, [0, 2, 1, 3]), run_time=run_time)
        self.update_ticker()
        self.wait(1)

        self.remove(self.ticker)
        for y in added_objects:
            self.remove(*y)

    def construct(self):
        nplane = manim.NumberPlane()
        self.add(nplane)

        # self.tick_value = 0
        # self.ticker = None
        # self.set_ticker()

        # count = 16
        # org_mat = np.arange(start=1, step=1, stop=count + 1).reshape((count // 4, -1))
        # walker = (2, 1)

        # mat3 = BestMatrix(org_mat, walker)
        # self.play_with(mat3)

        chain = MarkovChainThreeState()
        self.add(*chain.get_render_objects())
        self.wait(2)
