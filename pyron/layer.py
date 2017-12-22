from pyron import neuron as nn
import numpy as np

class Layer:
    def __init__(self, ID , units , actFn , model , input_dims , dropout):
        self.ID = ID
        self.units = units
        self.actFn = actFn
        self.neurons = self.getNN(units , actFn)
        self.input_ = []
        self.output = []
        self.input_dims = input_dims
        self.model = model
        self.dropout = dropout
        self.delta = []

    def getNN(self, units ,actFn):
        n = []
        for i in range(units):
            n.append(nn.Neuron(actFn , self))
        return n

    def Fire(self):
        for n in self.neurons:
            n.Fire()

    def setInput(self , x = []):
        print("ID: " , self.ID)
        if len(x) != 0:
            self.input_ = x
        elif len(x) == 0:
            self.model.layers[self.ID - 1].pushOutput()
            self.input_ = self.model.layers[self.ID - 1].output
        for n in self.neurons:
            n.setInput(self.input_)

    def pushOutput(self):
        out = []
        for n in range(len(self.neurons)):
            out.append(self.neurons[n].fire)
        self.output = np.squeeze(out)

    def gatherDelta(self):
        return [n.delta for n in self.neurons]

    def drop(self):
        pass

