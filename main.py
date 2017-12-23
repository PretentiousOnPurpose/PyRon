# For Deep Learning Algorithms

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

g = seq.layers[3].neurons[0].input_
w = seq.layers[3].neurons[0].weights

print(np.dot(g.T , w))
















# For Machine Learning Algorithms

# import numpy as np
# from pyron.ml import LinearRegression

# x = np.linspace(1 , 100 , 100)
# y = 2*x + 5

# from sklearn.model_selection import train_test_split
# xtr , xts , ytr , yts = train_test_split(x , y , test_size =0.05)

# from sklearn.preprocessing import MinMaxScaler
# scalerX = MinMaxScaler()
# scalerY = MinMaxScaler()

# scalerX.fit(x.reshape(-1, 1)); scalerY.fit(y.reshape(-1, 1))
# xtr = scalerX.transform(xtr.reshape(-1, 1)); xts = scalerX.transform(xts.reshape(-1, 1))
# ytr = scalerX.transform(ytr.reshape(-1, 1)); yts = scalerX.transform(yts.reshape(-1, 1))
 
# model = LinearRegression()
# model.fit(xtr  , ytr)