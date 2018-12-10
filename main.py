# Created by: Elias Mote, Justin Ramos

from Helpers import TestSimpleData, LoadInputs
from Helpers import TestSimpleData
from Problem1 import Problem1
from Problem2 import Problem2
from Problem3 import Problem3
from NeuralNetwork import TestNetwork
from Helpers import LoadInputs
from Problem1 import Problem1
from Problem2 import Problem2
from Problem3 import Problem3
from LoadMNISTDataSets import LoadData
import json
import numpy as np



def main():
	greyscale_range = 255
	
	# Load in the MNIST data set
	# x_train and x_test are uint8 arrays of grayscale image data with
	# shape (num_samples, 28, 28).
	# y_train and y_test are uint8 arrays of digit labels 
	# (integers in range 0-9) with shape (num_samples,).
	(x_train, y_train), (x_test, y_test) = LoadData()

	train_sets = LoadInputs(x_train, y_train)
	test_sets = LoadInputs(x_test, y_test)

	print("Problem 1")
	Problem1(train_sets[7], train_sets[9], test_sets[7], test_sets[9])

	print("Problem 2")
	best_multi_percept = Problem2(train_sets, test_sets)

	print("Problem 3")
	best_nrl_network = Problem3(x_train, y_train, x_test, y_test)

	# Final step: accept as input a single image of a hand written digit
	#file_digit = None
	#while(not file_digit.isnumeric()):
	file_digit = input("Enter number 0-9 for digit recognition: ")
	input_file = open("test" + str(file_digit) + ".json")


	# Input file should be structed as a single digit array.
	# Essentially, the same format as before but with only one digit as
	# only the first value in the array will be checked
	data = json.load(input_file)
	per_data = LoadInputs(np.array(data[0:1]),[int(file_digit)])

	# Test the selcted digit image file
	per_digit_guess = best_multi_percept.ActivationLabel(per_data[int(file_digit)][0])
	neu_digit_guess = TestNetwork(best_nrl_network["num_hidden_nodes"], best_nrl_network["hidden_wgts"],
		best_nrl_network["hidden_acts"], best_nrl_network["output_wgts"],
		best_nrl_network["output_acts"], data)

	# Report the guessed digit
	print("Multiclass perceptron: guessed digit is " + str(per_digit_guess))
	print("Neural network: guessed digit is " + str(neu_digit_guess))

if __name__ == '__main__':
	main()
