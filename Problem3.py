# Created by: Elias Mote, Justin Ramos

def Problem3(x_train, y_train, x_test, y_test):

	from NeuralNetwork import NeuralNetwork

	greyscale_range = 255
	neural_networks = []
	neural_networks.append(NeuralNetwork(x_train, y_train, x_test, y_test, greyscale_range, 3))
	neural_networks.append(NeuralNetwork(x_train, y_train, x_test, y_test, greyscale_range, 6))
	neural_networks.append(NeuralNetwork(x_train, y_train, x_test, y_test, greyscale_range, 10))
	neural_networks.append(NeuralNetwork(x_train, y_train, x_test, y_test, greyscale_range, 15))
	neural_networks.append(NeuralNetwork(x_train, y_train, x_test, y_test, greyscale_range, 21)) 

	# Report findings for optimal network
	best_network = neural_networks[0]
	for n in neural_networks:
		if(n["error_rate"] < best_network["error_rate"]):
			best_network = n

	print(str(best_network["num_hidden_nodes"]) + " hidden layer neurons gave the minimum error rate")
	print("Epoch with the minimum error is " + str(best_network["epoch_min_error"]))
	print("Error rate is " + str(best_network["error_rate"]))
	print("Hidden layer neuron weights:")
	print(best_network["hidden_wgts"])
	print("Output layer neuron weights:")
	print(best_network["output_wgts"])
	print("Confusion matrix:")
	print("X axis is the actual value")
	print("Y axis is the guessed value")
	print(best_network["confusion_matrix"])

	return best_network

	"""
	import json
	import numpy as np

	x_train, y_train, x_test, y_test = [],[],[],[]

	# Build the training files
	for i in range(10):

		# Get the data from the json file
		input_file = open("sample_handwriting_number.json")
		data = json.load(input_file)
		
		# When we are rebuilding the files, the x and y sets will be in order.
		# Thus, randomizing the input to the machine learning algorithms is important
		# Build the x file and y file
		for j in range(len(data)):
			x_train.append(data[j])
			y_train.append(i)

	# Build the test files
	for i in range(10):

		# Get the data from the json file
		input_file = open("sample_handwriting_number.json")
		data = json.load(input_file)
		
		# When we are rebuilding the files, the x and y sets will be in order.
		# Thus, randomizing the input to the machine learning algorithms is important
		# Build the x file and y file
		for j in range(len(data)):
			x_test.append(data[j])
			y_test.append(i)

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	a = Problem3(x_train, y_train, x_test, y_test, greyscale_range, best_prob3["num_hidden_nodes"])
	print("Digit guess is " + a["cur_guess"])
	"""