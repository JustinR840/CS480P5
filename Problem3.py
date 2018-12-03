import math
import numpy as np
import random as r
import tensorflow as tf
import datetime as dt
import sys

# Compute activation of hidden neurons
def hidden_act(input_set, hidden_wgts, hidden_acts, num_hidden_nodes, m):
	wgt_sums = np.dot(hidden_wgts, input_set[m])
	hidden_acts = (1 / (1 + np.exp(-1 * wgt_sums)))
	hidden_acts = np.append(hidden_acts, [1])
	return hidden_acts

# Compute activation of output neurons
def output_act(hidden_acts, output_wgts, output_acts, num_hidden_nodes, num_output_nodes):
	wgt_sums = np.dot(output_wgts, hidden_acts)
	output_acts = (1 / (1 + np.exp(-1 * wgt_sums)))
	return output_acts

# Compute error at the output layer neurons
def compute_output_error(output_acts, num_output_nodes, target, delta_o):
	delta_o = -1 * (output_acts - target) * output_acts * (1 - output_acts)
	return delta_o

# Compute error at the hidden layer neurons
def compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, delta_o, delta_h):
	delta_h = np.dot(delta_o,output_wgts) * hidden_acts * (1-hidden_acts)
	return delta_h

# Update output layer weights
def update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta):
	for i in range(num_output_nodes):
		for j in range(num_hidden_nodes):
			output_wgts[i][j] = (output_wgts[i][j] + eta * delta_o[i] * hidden_acts[j])
	return output_wgts

def update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m):
	for i in range(num_hidden_nodes):
		for j in range(len(input_set[0])):
			hidden_wgts[i][j] = (hidden_wgts[i][j] + eta * delta_h[i] * input_set[m][j])
	return hidden_wgts

def Problem3(x_train, y_train, x_test, y_test, greyscale_range, num_hidden_nodes):

	# Process start time
	start_time = dt.datetime.now()

	# Init the RNG
	r.seed()

	set_len = 20
	#set_len = len(x_train)

	# Prepare the training set
	train_set = [[0 for _ in range(49)] for _ in range(set_len)]
	#for i in range(len(x_train)):
	for i in range(set_len):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):

						# Divide by 16 for averaging the 4x4 block.
						# Also divide by 255 so each grayscale value varies from 0 to 1.
						# This helps keep the neural network from becoming oversaturated
						total += x_train[i][x][y] / 16 / 255
				train_set[i][j*7//4+k//4] = total

	# Prepare the test set
	test_set = [[0 for _ in range(49)] for _ in range(set_len)]
	for i in range(set_len):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):

						# Divide by 16 for averaging the 4x4 block.
						# Also divide by 255 so each grayscale value varies from 0 to 1.
						# This helps keep the neural network from becoming oversaturated
						total += x_test[i][x][y] / 16 / 255
				test_set[i][j*7//4+k//4] = total

	# Bias for the input and hidden node
	bias = 1

	# Append the bias to each input and test vector
	for i in range(len(train_set)):
		train_set[i].append(bias)
	for i in range(len(test_set)):
		test_set[i].append(bias)

	train_set = np.array(train_set, dtype=np.float64)
	test_set = np.array(test_set, dtype=np.float64)

	# Create a target weight vector for our training set
	old_targets = y_train.tolist()[0:set_len]
	targets = []
	for i in range(len(old_targets)):
		t = []
		for j in range(10):
			if(j == old_targets[i]):
				t.append(1)
			else:
				t.append(0)
		targets.append(t)

	targets = np.array(targets, dtype=np.float64)

	# Create a target weight vector for our test set
	old_tests = y_test.tolist()[0:set_len]
	tests = []
	for i in range(len(old_tests)):
		t = []
		for j in range(10):
			if(j == old_tests[i]):
				t.append(1)
			else:
				t.append(0)
		tests.append(t)

	tests = np.array(tests, dtype=np.float64)

	# The number of output nodes
	num_output_nodes = len(targets[0])

	# Initial setup for the hidden node and output node weights
	hidden_wgts = [[0.1 for _ in range(len(train_set[0]))] for _ in range(num_hidden_nodes)]
	hidden_wgts = np.array(hidden_wgts, dtype=np.float64)
	output_wgts = [[0.1 for _ in range(num_hidden_nodes+1)] for _ in range(num_output_nodes)]
	output_wgts = np.array(output_wgts, dtype=np.float64)
	num_epochs = 1000

	# Initialize activations
	hidden_acts = np.zeros(num_hidden_nodes+1, dtype=np.float64)
	hidden_acts[num_hidden_nodes] = 1
	output_acts = np.zeros(num_output_nodes+1, dtype=np.float64)
	output_acts[num_output_nodes] = 1

	# Initialize delta error
	delta_h = np.zeros(num_hidden_nodes, dtype=np.float64)
	delta_o = np.zeros(num_output_nodes, dtype=np.float64)

	# Learning coefficient
	eta = 1

	# Run the training algorithm for a number of epochs
	for n in range(num_epochs):

		# Run the training set in a randomized order
		train_order = []
		for i in range(len(train_set)):
			train_order.append(i)
		#r.shuffle(train_order)

		# Run the training set
		for m in range(len(train_set)):
			hidden_acts = hidden_act(train_set, hidden_wgts, hidden_acts, num_hidden_nodes, train_order[m])
			output_acts = output_act(hidden_acts, output_wgts, output_acts, num_hidden_nodes,num_output_nodes)
			delta_o = compute_output_error(output_acts, num_output_nodes, targets[train_order[m]], delta_o)
			delta_h = compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, delta_o, delta_h)
			output_wgts = update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta)
			hidden_wgts = update_hidden_wgts(train_set, num_hidden_nodes, hidden_wgts, delta_h, eta, train_order[m])

		if(n % (num_epochs // 10) == 0):
			print(str(n) + " epochs have been completed.")

	# Keep track of how many times the neural network has guessed correctly
	num_successes = 0

	# The confusion matrix
	confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]

	# Run the test set in a randomized order
	test_order = []
	for i in range(len(test_set)):
		test_order.append(i)
	#r.shuffle(test_order)

	# Run the test set
	for m in range(len(test_set)):

		# The value we want
		target = y_test[test_order[m]]

		# Recall algorithm
		hidden_acts = hidden_act(test_set, hidden_wgts, hidden_acts, num_hidden_nodes, test_order[m])
		output_acts = output_act(hidden_acts, output_wgts, output_acts, num_hidden_nodes,num_output_nodes)

		# Which value is largest, aka, the algo's current guess?
		max_val = -999999
		cur_guess = -1

		# Find the max value in the current vector
		# If the current value is larger, mark that value as our current guess
		#print(output_acts)
		for i in range(num_output_nodes):
			if(output_acts[i] > max_val):
				max_val = output_acts[i]
				cur_guess = i

		# If the guess is correct, mark it as a success
		if(target == cur_guess):
			num_successes += 1

		# Increment the confusion matrix
		#print("target = " + str(target))
		#print("cur_guess = " + str(cur_guess))
		#print()
		confusion_matrix[target][cur_guess] += 1

	# Report our findings
	delta_time = (dt.datetime.now() - start_time)
	print("The process took " + str(delta_time))
	print("There were " + str(num_successes) + " sucesses.")
	print(str(num_successes * 100 / len(test_set)) + "% are correct.")
	print("Confusion matrix:")
	print("X axis is the guessed value")
	print("Y axis is the actual value")
	print(confusion_matrix)