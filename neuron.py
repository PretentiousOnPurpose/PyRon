import numpy as np

class Neuron:
    def __init__(self, actFn , layer):
        self.actFn = actFn
        self.input = np.array([])
        self.weights = np.array([])
        self.bias = np.random.random(1)[0]
        self.layer = layer

    def setInput(self , x):
        self.input = x

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

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def gradSigmoid(self, x):
        val = self.sigmoid(x)
        return val*(1 - val)

    def gradRELU(self , x):
        pass

    def gradLinear(self , x):
        pass

