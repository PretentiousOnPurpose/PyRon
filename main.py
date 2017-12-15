from sequential import Sequential

Seq = Sequential()
Seq.add(1 , "linear" , input_dims=1)
Seq.add(3 , "relu")
Seq.add(3 , "relu")
Seq.add(3 , "relu")
Seq.add(1 , "relu")
Seq.compile()
Seq.describe()

