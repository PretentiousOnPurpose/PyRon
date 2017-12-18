import numpy as np
from pyron.sequential import Sequential


x = np.linspace(1 , 5 , 5).reshape(5 ,)
y = 2*x

seq = Sequential()
seq.add(1 , "linear" , input_dims=1)
seq.add(1 , "relu")
seq.compile(loss="mean_squared_error" , optimiser= "sgd")
seq.train(x ,y ,0.05, 1)


print(seq.layers[0].output)
# Check Predict Function