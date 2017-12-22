import numpy as np

class Neuron:
    def __init__(self, actFn , layer):
        self.actFn = actFn
        self.input_ = np.array([])
        self.weights = np.array([])
        self.bias = np.random.random(1)[0]
        self.layer = layer
        self.state = True
        self.delta = np.array([])
        self.fire = np.array([])
        self.pot = np.array([])

    def setInput(self , x):
        self.input_ = x

    def Pot(self):
        print(self.input_)
        print(self.weights)
        self.pot = (self.weights * self.input_) + self.bias
        return self.pot

    def Fire(self):
        if self.state:
            pot = self.Pot()
            if self.actFn.lower() == "relu":
                self.fire = [k if k > 0 else 0 for k in pot]
            elif self.actFn.lower() == "sigmoid":
                self.fire = 1 / (1 + np.exp(-pot))
            elif self.actFn.lower() == "linear":
                self.fire = pot
        else:
            self.fire = np.zeros(len(self.input_))
        return self.fire

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def grad(self):
        if self.actFn.lower() == "sigmoid":
            return self.fire*(1 - self.fire)
        elif self.actFn.lower() == "relu":
            if self.pot < 0:
                return 0
            else:
                return 1


