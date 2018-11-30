import math
import numpy as np
from copy import deepcopy
import random as r
import tensorflow as tf

# Test each component of the multilayer perceptron individually
# Compute activation of hidden neurons
def hidden_act(input_set, hidden_wgts, num_hidden_nodes, m):
	hidden_acts = []
	for i in range(num_hidden_nodes):
		weighted_sum = np.dot(input_set[m], hidden_wgts[i])
		hidden_acts.append(1 / (1 + math.exp(-1 * weighted_sum)))
	return hidden_acts

# Compute activation of output neurons
def output_act(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes):
	hidden = deepcopy(hidden_acts)
	bias = 1
	hidden.append(bias)
	output_acts = []
	for i in range(num_output_nodes):
		weighted_sum = np.dot(hidden, output_wgts[i])
		output_acts.append(1 / (1 + math.exp(-1 * weighted_sum)))
	return output_acts

def compute_output_error(output_acts, num_output_nodes, target):
	delta_o = []
	for i in range(num_output_nodes):
		delta_o.append(-1 * (output_acts[i] - target[i]) * output_acts[i] * ((1 - output_acts[i])))
	return delta_o

# Compute error at the hidden layer neurons
def compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, delta_o):
	delta_h = []
	for i in range(num_hidden_nodes):
		delta_h.append(delta_o[0] * output_wgts[0][i] * hidden_acts[i] * (1-hidden_acts[i]))
	return delta_h

# Update output layer weights
def update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta):
	new_output_wgts = deepcopy(output_wgts)
	new_hidden_acts = deepcopy(hidden_acts)
	new_hidden_acts.append(1)
	for i in range(num_output_nodes):
		for j in range(3):
			new_output_wgts[i][j] = (new_output_wgts[i][j] + eta * delta_o[i] * new_hidden_acts[j])
	return new_output_wgts

def update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m):
	new_hidden_wgts = deepcopy(hidden_wgts)
	for i in range(num_hidden_nodes):
		for j in range(len(input_set[0])):
			new_hidden_wgts[i][j] = (hidden_wgts[i][j] + eta * delta_h[i] * input_set[m][j])
	return new_hidden_wgts

def run_perceptron(x_train, y_train, x_test, y_test, num_hidden_nodes):

	set_len = 5
	#set_len = len(x_train)
	input_set = [[0 for _ in range(49)] for _ in range(set_len)]
	#for i in range(len(x_train)):
	for i in range(set_len):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_train[i][x][y] / 16 / 255
				input_set[i][j*7//4+k//4] = total


	test_set = [[0 for _ in range(49)] for _ in range(set_len)]
	for i in range(set_len):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_test[i][x][y] / 16 / 255
				test_set[i][j*7//4+k//4] = total



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

	#num_hidden_nodes = 5
	num_output_nodes = len(targets[0])
	hidden_wgts = [[r.uniform(0,1) for _ in range(len(input_set[0])+1)] for _ in range(num_hidden_nodes)]
	output_wgts = [[r.uniform(0,1) for _ in range(num_hidden_nodes+1)] for _ in range(num_output_nodes)]
	num_epochs = 1000
	eta = 1
	bias = 1

	for i in range(len(input_set)):
		input_set[i].append(bias)
	for i in range(len(test_set)):
		test_set[i].append(bias)
	for n in range(num_epochs):
		for m in range(len(input_set)):
			hidden_acts = hidden_act(input_set, hidden_wgts, num_hidden_nodes, m)
			output_acts = output_act(hidden_acts, output_wgts, num_hidden_nodes,num_output_nodes)
			delta_o = compute_output_error(output_acts, num_output_nodes, targets[m])
			delta_h = compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, delta_o)
			output_wgts = update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta)
			hidden_wgts = update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m)

		if(n % (num_epochs // 20) == 0):
			print(str(n) + " epochs have been completed.")

	for m in range(len(test_set)):
		print(tests[m])
		hidden_acts = hidden_act(test_set, hidden_wgts, num_hidden_nodes, m)
		output_acts = output_act(hidden_acts, output_wgts, num_hidden_nodes,num_output_nodes)
		print(output_acts)