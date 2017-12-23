import numpy as np

class LinearRegression():
    def __init__(self):
        self.data = np.array([])
        self.target = np.array([])
        self.weights = np.array([])
        self.bias = np.random.random(1).reshape(1 , 1)
        self.input_dims = 0

    def fit(self , x, y , rate=0.05):
        self.data = x; self.target = y
        self.weights = np.random.random(len(x[0]))
        loss = 10
        while loss > 0.009:
            y_ = x*self.weights + self.bias
            err = (y_ - y)
            loss = np.sum(err, axis = 1, keepdims=True)**2/2*len(x)
            self.weights += rate*np.mean(np.sum(err*x , axis=1 , keepdims=True))
            self.bias += rate*np.mean(err, axis=1, keepdims=True)

    def predict(self, v):
        if len(self.data[0]) != len(v):
            print("Dimension Error for one sample")
            return

        return self.weights*v + self.bias

class LogisticRegression():
    def __init__(self):
        self.data = np.array([])
        self.target = np.array([])
        self.weights = np.array([])
        self.bias = np.random.random(1)
        self.input_dims = 0

    def fit(self, x, y, rate=0.05):
        self.data = x;
        self.target = y
        self.weights = np.random.random(len(x[0]))
        loss = 10
        while loss > 0.009:
            y_ = x * self.weights + self.bias
            err = (y_ - y)
            loss = np.sum(err, axis=1, keepdims=True) ** 2 / 2 * len(x)
            self.weights += rate * np.mean(np.sum(err * x, axis=1, keepdims=True))
            self.bias += rate * np.mean(err, axis=1, keepdims=True)

    def predict(self, v):
        if len(self.data[0]) != len(v):
            print("Dimension Error for one sample")
            return
        return 1 / (1 + np.exp(- (self.weights * v + self.bias)))


class SVMClassifier():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass

class SVMRegressor():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass

class kNNClassifier():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass

class RBF():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass

class adaboost():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass

class PCA():
    def __init__(self):
        self.data = []
        self.target = []
        self.weights = []
        self.bias = 0
        self.input_dims = 0

    def fit(self, x, y, norm=True):
        pass
