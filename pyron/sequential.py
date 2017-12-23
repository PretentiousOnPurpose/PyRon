import numpy as np

from pyron import layer as l


class Sequential:
    def __init__(self):
        self.layers = []
        self.ID = 0
        self.optimiser = ""
        self.loss = ""

    def add(self, units ,actFn , dropout=0.0,  input_dims=None):
        self.layers.append(l.Layer(self.ID , units , actFn , self , input_dims, dropout))
        self.ID += 1

    def predict(self, x):
        self.layers[0].setInput(x)
        self.layers[0].Fire()
        self.layers[0].pushOutput()
        for i in range(len(self.layers) - 1):
            self.layers[i+1].setInput()
            self.layers[i+1].Fire()
            self.layers[i+1].pushOutput()

        # return self.layers[-1].output
        return np.squeeze(np.ones((len(x), 1))*0.54)

    def feed(self , x):
        for n in range(len(self.layers[0].neurons)):
            # self.layers[0].neurons[n].weights = np.random.random(self.layers[0].input_dims)
            self.layers[0].neurons[n].weights = np.ones((self.layers[0].input_dims , len(self.layers[0].neurons)))

        self.layers[0].setInput(x)
        self.layers[0].Fire()
        self.layers[0].pushOutput()

        self.layers[1].setInput()
        self.layers[1].Fire()
        self.layers[1].pushOutput()

        self.layers[2].setInput()
        self.layers[2].Fire()
        self.layers[2].pushOutput()

        self.layers[3].setInput()
        

        # for l in self.layers[1:]:
        #     l.setInput()
        #     l.Fire()
        #     l.pushOutput()

        return self.layers[-1].output


    def compile(self, loss , optimiser):
        self.loss = loss
        self.optimiser = optimiser
        for l in range(1, len(self.layers)):
            for n in self.layers[l].neurons:
                # n.weights = np.random.random((len(self.layers[l].neurons) , len(self.layers[l-1].neurons)))
                n.weights = np.ones((self.layers[l-1].units , 1))

    def test(self, x):
        self.predict(x)

    def train(self , x, y , rate , epochs):
        for n in range(len(self.layers[0].neurons)):
            # self.layers[0].neurons[n].weights = np.random.random(self.layers[0].input_dims)
            self.layers[0].neurons[n].weights = np.ones((self.layers[0].input_dims , len(self.layers[0].neurons)))
        for e in range(epochs):
            y_ = self.predict(x)
            err = y - y_
            loss = self.getLoss(y , y_)
            print("Iter: " , e ," | Loss: " , loss)
            self.backProp(err, rate)

    def backProp(self, err , rate):
        if self.optimiser == "sgd":
            if self.loss == "binary_cross_entropy":
                for n in self.layers[-1].neurons:
                    delta = rate*np.sum(err , axis= 1 ,keepdims=True)/len(err)
                    n.delta = delta
                    n.weights = n.weights + delta*n.input_
                self.layers[-1].gatherDelta()

                for la in self.layers[-2: 0:-1]:
                    for n in la.neurons:
                        n.delta = np.dot(n.weights , self.layers[la.ID + 1].delta)
                        n.weights = n.weights + n.delta*n.grad()*n.input_
                    la.gatherDelta()

            elif self.loss == "multi_cross_entropy":
                pass
            elif self.loss == "mean_squared_error":
                for n in self.layers[-1].neurons:
                    delta = float(rate*np.mean(err)/len(err))
                    n.delta = delta
                    n.weights = n.weights + delta*n.input_
                self.layers[-1].gatherDelta()

                for la in self.layers[-2: 0:-1]:
                    for n in la.neurons:
                        n.delta = np.dot(n.weights , self.layers[la.ID + 1].delta)
                        n.weights = n.weights + n.delta*n.grad()*n.input_
                    la.gatherDelta()

        elif self.optimiser == "adam":
            pass
        elif self.optimiser == "adagrad":
            pass

    def describe(self):
        print("----Model Description----")
        print("Loss: ", self.loss , " | Optimiser: ",  self.optimiser)
        print("Hidden Layers: " , [0 if len(self.layers)-2 <= 0 else len(self.layers)-2][0])
        print("----------------------------")
        print("Input Layer: " , "Units: ", self.layers[0].units , "| ActFn: " , self.layers[0].actFn , " | Weights: " , self.layers[0].neurons[0].weights.shape, " | Input_dims: ", self.layers[0].input_dims)
        for l in self.layers[1:-1]:
            print("HL " , l.ID , ": Units: " , l.units , "| ActFn: " , l.actFn , " | Weights: " , l.neurons[0].weights.shape)
        print("Output Layer: ", "Units: ", self.layers[-1].units , "| ActFn: " , self.layers[-1].actFn , " | Weights: " , self.layers[-1].neurons[0].weights.shape)

    def getLoss(self, true , pred):
        if self.loss == "binary_cross_entropy":
            return np.sum(np.sum(-(true*np.log(pred) + (1 - true)*np.log(1 - pred)) , axis=1 , keepdims=True)/len(true))
        elif self.loss == "mean_squared_error":
            return np.mean(0.5*((true - pred)**2))
        elif self.loss == "multi_cross_entropy":
            return 1