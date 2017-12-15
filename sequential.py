import layer as l
import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.ID = 0

    def add(self, units ,actFn , input_dims=None):
        self.layers.append(l.Layer(self.ID , units , actFn , self , input_dims))
        self.ID += 1

    def predict(self, x):
        self.layers[0].input = x
        self.layers[0].fire()
        self.layers[0].pushOutput()
        for i in range(len(self.layers) - 1):
            self.layers[i+1].setInput()
            self.layers[i+1].fire()
            self.layers[i+1].pushOutput()
        return self.layers[-1].output

    def compile(self):
        for l in range(1, len(self.layers)):
            units = self.layers[l-1].units
            for n in self.layers[l].neurons:
                n.weights = np.random.random(units)

    def train(self , x, y , rate , epochs):
        for i in range(self.layers[0].input_dims):
            for n in range(self.layers[0].neurons):
                self.layers[0].neurons[n] = np.random.random(self.layers[0].input_dims)
        for e in range(epochs):
            y_ = self.predict(x)
            err = y_ - y
            self.optimise(x, y , err, rate)

    def optimise(self,x ,y , err, rate):
        for l in self.layers[1:]:
            pass

    def backProp(self, err , rate):
        pass

    def describe(self):
        print("----Model Description----")
        print("Hidden Layers: " , len(self.layers)-2)
        print("----------------------------")
        print("Input Layer: " , "Units: ", self.layers[0].units , "| ActFn: " , self.layers[0].actFn , " | Weights: " , self.layers[0].neurons[0].weights.shape, " | Input_dims: ", self.layers[0].input_dims)
        for l in self.layers[1:-1]:
            print("HL " , l.ID , ": Units: " , l.units , "| ActFn: " , l.actFn , " | Weights: " , l.neurons[0].weights.shape)
        print("Output Layer: ", "Units: ", self.layers[-1].units , "| ActFn: " , self.layers[-1].actFn , " | Weights: " , self.layers[-1].neurons[0].weights.shape)