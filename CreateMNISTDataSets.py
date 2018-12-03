import tensorflow as tf
import json

def CreateFiles():

	# Get the MNIST
	mnist = tf.keras.datasets.mnist

	# Load in the MNIST data set
	# x_train and x_test are uint8 arrays of grayscale image data with
	# shape (num_samples, 28, 28).
	# y_train and y_test are uint8 arrays of digit labels 
	# (integers in range 0-9) with shape (num_samples,).
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.tolist()
	y_train = y_train.tolist()
	#x_test = x_test.tolist()
	#y_test = y_test.tolist()

	# Get the indices of each handwritten number 0-9
	training_labels = [[] for _ in range(10)]
	for i in range(len(y_train)):
	#for i in range(len(y_test)):
		training_labels[y_train[i]].append(i)
		#training_labels[y_test[i]].append(i)

	# Create a two-dimensional array that stores the handwritten data by number
	training_data = [[] for _ in range(10)]
	for i in range(10):
		for j in range(len(training_labels[i])):
			training_data[i].append(x_train[training_labels[i][j]])
			#training_data[i].append(x_test[training_labels[i][j]])

	# Convert the handwritten data to json objects, then dump them to files
	data_encode = [None for _ in range(10)]
	for i in range(10):
		data_encode[i] = json.dumps(training_data[i])
		data = json.loads(data_encode[i])
		with open('train' + str(i) + '.json', 'w') as outfile:
		#with open('test' + str(i) + '.json', 'w') as outfile:
		    json.dump(data, outfile)