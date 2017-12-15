import neuron as nn

class Layer:
    def __init__(self, ID , units , actFn , model , input_dims):
        self.ID = ID
        self.units = units
        self.actFn = actFn
        self.neurons = self.getNN(units , actFn)
        self.input = []
        self.output = []
        self.input_dims = input_dims
        self.model = model

    def getNN(self, units ,actFn):
        n = []
        for i in range(units):
            n.append(nn.Neuron(actFn , self))
        return n

    def fire(self):
        for n in range(len(self.neurons)):
            self.neurons[n].fire()

    def setInput(self):
        self.model.layers[self.ID-1].pushOutput()
        self.input = self.model.layers[self.ID-1].output
        for n in self.neurons:
            n.setInput(self.input)

    def pushOutput(self):
        out = []
        for n in range(len(self.neurons)):
            out.append(self.neurons[n].output)
        self.output = out


