import numpy as np


class MemorySimple:

    def __init__(self, _capacity):
        self.data = []

    def store(self, experience):
        self.data.append(experience)

    def __getitem__(self, item):
        data = self.data[item]
        return data

    def sample(self):
        return np.random.choice(self.data)

    def __iter__(self):
        return iter(self.data)
