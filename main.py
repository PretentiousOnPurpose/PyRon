import numpy as np
from pyron.sequential import Sequential

x = np.linspace(1 , 100 , 100).reshape(100 , 1)
y = 2*x + 5

seq = Sequential()
seq.add(1 , "linear" , input_dims=1)
seq.add(3 , "relu")
seq.add(3 , "relu")
seq.add(3 , "relu")
seq.add(1 , "relu")
seq.compile(loss="binary_cross_entropy" , optimiser= "sgd")
# seq.train(x, y ,0.05, 150)

# Check Predict Method
