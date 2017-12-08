import layer as l

class Sequential:
    def __init__(self):
        self.layers = []
        self.ID = 0



    def add(self,ID, units ,actFn , input_dims=None):
        self.layers.append(l.Layer(units , actFn , input_dims))
        self.ID += 1

    def feedforward(self, x):
        self.layers[0].input = x
        self.layers[0].fire()
        self.layers[0].pushOutput()
        for i in range(len(self.layers) - 1):
            self.layers[i+1].setInput()
            self.layers[i+1].fire()
            self.layers[i+1].pushOutput()
        return self.layers[-1].output
