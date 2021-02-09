import numpy as np
from ..metas import ReplayMemory


class MemorySimple(ReplayMemory):

    def __init__(self, _capacity):
        self.data = []

    def store(self, experience):
        self.data.append(experience)

    def __getitem__(self, item):
        data = self.data[item]
        return data

    def sample(self, batch_size=1):
        return np.random.choice(self.data, size=batch_size)

    def __iter__(self):
        return iter(self.data)

    def clear(self):
        self.data = []
