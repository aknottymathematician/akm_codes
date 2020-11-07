import tensorflow as tf
import numpy as np
import time



# num_features = 784

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])


from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

#print(y_train)

class DeepNeuralNetwork():
	def __init__(self, sizes, epochs=1, l_rate=0.001):
		self.sizes = sizes
		self.epochs = epochs
		self.l_rate = l_rate

		# we save all parameters in the neural network in this dictionary
		self.params = self.initialization()


	def sigmoid(self, x, derivative=False):
		if derivative:
			return (np.exp(-x))/((np.exp(-x)+1)**2)
		return 1/(1 + np.exp(-x))

	def softmax(self, x, derivative=False):
		# Numerically stable with large exponentials
		exps = np.exp(x - x.max())
		if derivative:
			return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
		return exps / np.sum(exps, axis=0)

	def initialization(self):
		# number of nodes in each layer
		input_layer=self.sizes[0]
		#print("INPUT LAYER",input_layer)
		hidden_1=self.sizes[1]
		hidden_2=self.sizes[2]
		output_layer=self.sizes[3]

		params = {
			'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
			'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
			'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer),
			'B1':np.ones(128, dtype='int')*0.1,
			'B2':np.ones(64, dtype='int')*0.1,
			'B3':np.ones(10, dtype='int')*0.1
		}
		#print(params['W2'].shape)
		return params

	def feedforward(self, x_train):
		params = self.params

		# input layer activations becomes sample
		params['A0'] = x_train
		#print(x_train.shape)
		# input layer to hidden layer 1 and passed through activation function
		params['Z1'] = np.dot(params["W1"], params['A0']) + params['B1']
		params['A1'] = self.sigmoid(params['Z1'])

		# hidden layer 1 to hidden layer 2 and passed through activation function
		params['Z2'] = np.dot(params["W2"], params['A1']) + params['B2']
		params['A2'] = self.sigmoid(params['Z2'])

		# hidden layer 2 to output layer
		params['Z3'] = np.dot(params["W3"], params['A2'])+params['B3']
		params['A3'] = self.softmax(params['Z3'])
		#print(params['A3'].shape)
		return params['A3']

	def backpropagation(self, y_train, output):

		params = self.params
		change_w = {}

		# Calculate W3 update
		error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
		change_w['W3'] = np.outer(error, params['A2'])

		# Calculate W2 update
		error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
		change_w['W2'] = np.outer(error, params['A1'])

		# Calculate W1 update
		error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
		change_w['W1'] = np.outer(error, params['A0'])

		return change_w

	def update_weights(self, changes_to_w):

		for key, value in changes_to_w.items():
			#print(key, value.shape)
			self.params[key] -= self.l_rate * value

	def accuracy(self, x_val, y_val):
		predictions = []

		for x, y in zip(x_val, y_val):
			output = self.feedforward(x)
			pred = np.argmax(output)
			predictions.append(pred == np.argmax(y))
		
		return np.mean(predictions)

	def train(self, x_train, y_train, x_val, y_val):
		start_time = time.time()
		for iteration in range(self.epochs):
			for x,y in zip(x_train, y_train):
				output = self.feedforward(x)
				#print("OUTPUT", output.shape)
				changes_to_w = self.backpropagation(y, output)
				self.update_weights(changes_to_w)
			
			accuracy = self.accuracy(x_val, y_val)
			print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
				iteration+1, time.time() - start_time, accuracy * 100
			))



dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_test, y_test)