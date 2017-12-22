import numpy as np
from pyron.sequential import Sequential


x = np.linspace(1 , 5 , 5).reshape(5 , 1)
y = 2*x

seq = Sequential()
seq.add(1 , "linear" , input_dims=1)
seq.add(4 , "relu")
seq.add(2 , "relu")
seq.add(1 , "relu")
seq.compile(loss="mean_squared_error" , optimiser= "sgd")
seq.feed(x)


print(seq.layers[1].output)
# h = seq.layers[1].neurons[2].input_
# w = seq.layers[1].neurons[2].weights
# print(h*w)