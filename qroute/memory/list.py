import numpy as np
from ..metas import ReplayMemory


class MemorySimple(ReplayMemory):

    def __init__(self, _capacity):
        self.data = []

    def store(self, *args):
        self.data.append(args)

    def __getitem__(self, item):
        return self.data[item]

    def sample(self, batch_size=1):
        return np.random.choice(self.data, size=batch_size)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def clear(self):
        self.data = []
