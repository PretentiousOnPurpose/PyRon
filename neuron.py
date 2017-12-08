import numpy as np

class Neuron:
    def __init__(self, actFn):
        self.actFn = actFn
        self.input = np.array([])
        self.weights = np.array([])
        self.bias = 0

    def setInput(self , x):
        self.input = x
        self.weights = np.random.random(len(x[0]))
        self.bias = np.random.random(1)

    def pot(self):
        return np.dot(self.weights.T , self.input) + self.bias

    def fire(self):
        pot = self.pot()
        if self.actFn.lower() == "relu":
            if pot > 0:
                return pot
            else:
                return 0
        elif self.actFn.lower == "sigmoid":
            return 1/(1 + np.exp(-self.input))

