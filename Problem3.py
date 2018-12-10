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