class MemorySimple:

    def __init__(self, _capacity):
        self.data = []
        self.error = []
        self.pointer = 0

    def store(self, experience):
        self.data.append(experience)
        self.error.append(-1)

    def sample(self, n):
        minibatch = self.data[self.pointer:self.pointer+n]
        b_idx = self.pointer
        b_is_weights = [1 for _ in minibatch]
        return b_idx, minibatch, b_is_weights

    def batch_update(self, tree_idx, abs_errors):
        for idx, error in enumerate(abs_errors):
            self.error[tree_idx + idx] + error
