import numpy as np

from pyron.sequential import Sequential

x = np.linspace(1 , 100 , 100)
y = 2*x + 5

Seq = Sequential()
Seq.add(1 , "linear" , input_dims=1)
Seq.add(3 , "relu")
Seq.add(3 , "relu")
Seq.add(3 , "relu")
Seq.add(1 , "relu")
Seq.compile(loss="binary_cross_entropy" , optimiser= "sgd")
Seq.train(x, y ,0.05, 150)

