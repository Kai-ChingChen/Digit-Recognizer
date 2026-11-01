from Value import Value
import numpy as np
class MLP:
    def __init__(self, seed = 42):
        init = np.random.default_rng(seed)
        self.W1 = Value(init.normal(size = (784, 64)) * np.sqrt(1/784))
        self.b1 = Value(np.zeros(64))
        self.W2 = Value(init.normal(size = (64, 64)) * np.sqrt(1/64))
        self.b2 = Value(np.zeros(64))
        self.W3 = Value(init.normal(size = (64, 64)) * np.sqrt(1/64))
        self.b3 = Value(np.zeros(64))
        self.W4 = Value(init.normal(size = (64, 10)) * np.sqrt(1/64))
        self.b4 = Value(np.zeros(10))

    def __call__(self, inputs):
        h = (inputs @ self.W1 + self.b1).tanh()
        h = (h @ self.W2 + self.b2).tanh()
        h = (h @ self.W3 + self.b3).tanh()
        return h @ self.W4 + self.b4

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4]