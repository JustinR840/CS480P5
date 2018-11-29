"""from Helpers import MultiLayerPerceptron


train = [[0,0],[0,1],[1,0],[1,1]]
targets = [0,0,0,1]

and_network = MultiLayerPerceptron(train,  1, 2, 1000, targets, train, targets)
print(and_network.GetTestOutputError())"""

import math
import numpy as np
from copy import deepcopy
import random as r

# Test each component of the multilayer perceptron individually
# Compute activation of hidden neurons
def hidden_act(input_set, hidden_wgts, num_hidden_nodes, m):
	hidden_acts = []
	for i in range(num_hidden_nodes):
		weighted_sum = np.dot(input_set[m], hidden_wgts[i])
		hidden_acts.append(1 / (1 + math.exp(-1 * weighted_sum)))
	return hidden_acts

def output_act(hidden_acts, num_hidden_nodes, num_output_nodes):
	output_acts = []
	for i in range(num_output_nodes):
		weighted_sum = np.dot(hidden_acts, output_wgts[i])
		output_acts.append(1 / (1 + math.exp(-1 * weighted_sum)))
	return output_acts

def compute_output_error(output_acts, num_output_nodes, target):
	delta_o = []
	for i in range(num_output_nodes):
		delta_o.append((output_acts[i] - target) * output_acts[i] * ((1 - output_acts[i])))
	return delta_o

# Compute error at the hidden layer neurons
def compute_hidden_error(hidden_acts, hidden_wgts, num_hidden_nodes, num_output_nodes, delta_o):
	delta_h = []
	for i in range(num_hidden_nodes):
		weighted_sum = np.dot(hidden_wgts, delta_o[0])
		delta_h.append(hidden_acts[i] * (1 - hidden_acts[i]) * weighted_sum)
	return delta_h

# Update output layer weights
def update_output_wgts(num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta):
	new_output_wgts = deepcopy(output_wgts)
	for i in range(num_output_nodes):
		for j in range(num_hidden_nodes):
			#new_output_wgts[i][j] = (output_wgts[i][j] - eta * delta_o[i] * hidden_acts[j])
			print(output_wgts[i][j])
			new_output_wgts[i][j] = (output_wgts[i][j] - delta_o[i])
			#print(new_output_wgts[i][j])
	return new_output_wgts

def update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m):
	new_hidden_wgts = deepcopy(hidden_wgts)
	for i in range(num_hidden_nodes):
		for j in range(len(input_set[0])):
			hidden_wgts[i][j] = (hidden_wgts[i][j] - eta * delta_h[i][j] * input_set[m][j])
	return new_hidden_wgts

input_set = [[0,0],[0,1],[1,0],[1,1]]
targets = [0,0,0,1]
num_hidden_nodes = 3
num_output_nodes = 1
hidden_wgts = [[0.1 for _ in range(len(input_set[0]))] for _ in range(num_hidden_nodes)]
output_wgts = [[0.1 for _ in range(num_hidden_nodes)] for _ in range(num_output_nodes)]
eta = 0
for _ in range(1):
	for m in range(2):
	#for m in range(len(input_set)):

		hidden_acts = hidden_act(input_set, hidden_wgts, num_hidden_nodes, m)
		#print(hidden_acts)
		output_acts = output_act(hidden_acts,num_hidden_nodes,num_output_nodes)
		#print(output_acts)
		delta_o = compute_output_error(output_acts, num_output_nodes, targets[m])
		#print(delta_o)
		delta_h = compute_hidden_error(hidden_acts, hidden_wgts, num_hidden_nodes, num_output_nodes, 
			delta_o)
		#print(delta_h)

		output_wgts = update_output_wgts(num_hidden_nodes, num_output_nodes, output_wgts, delta_o, 
			eta)
		#print(output_wgts)
		hidden_wgts = update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m)
	#print()

for m in range(len(input_set)):

	#print(input_set[m])
	hidden_acts = hidden_act(input_set, hidden_wgts, num_hidden_nodes, m)
	#print(hidden_acts)
	output_acts = output_act(hidden_acts,num_hidden_nodes,num_output_nodes)
	#print(output_acts)
	#print(compute_output_error(output_acts, num_output_nodes, target_wgts))
	delta_o = compute_output_error(output_acts, num_output_nodes, targets[m])
	#print(delta_o)
	#print()

