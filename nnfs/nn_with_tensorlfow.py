'''
FEEDFORWARD
input > weight > hidden layer 1 > activation function > weights > hidden layer 2 > activation functions > output layer

Compare output to intented output > Cost Function(RMSE)

Optimization Function > minimize the cost (Adam, AdaGrad, SGD)

Adjust the weights > Backpropagation

Feedforward + Backprogation = One Epoch!

'''


import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


#Architecture of the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#Parameters for training the model
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


model.save("basic_tf_mnist.model")