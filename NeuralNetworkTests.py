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

# Compute activation of output neurons
def output_act(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes):
	hidden = deepcopy(hidden_acts)
	bias = 1
	hidden.append(bias)
	#print(hidden)
	output_acts = []
	for i in range(num_output_nodes):
		weighted_sum = np.dot(hidden, output_wgts[i])
		output_acts.append(1 / (1 + math.exp(-1 * weighted_sum)))
	#print(output_acts)
	return output_acts

def compute_output_error(output_acts, num_output_nodes, target):
	delta_o = []
	for i in range(num_output_nodes):
		delta_o.append(-1*(output_acts[i] - target) * output_acts[i] * ((1 - output_acts[i])))
	return delta_o

# Compute error at the hidden layer neurons
def compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, delta_o):
	delta_h = []
	#print(delta_o)
	#print(output_wgts)
	#print(hidden_acts)
	for i in range(num_hidden_nodes):
	#print(delta_o[0] * output_wgts[0][0] * hidden_acts[0] * (1-hidden_acts[0]))
	#print(delta_o[0] * output_wgts[0][1] * hidden_acts[1] * (1-hidden_acts[1]))
		delta_h.append(delta_o[0] * output_wgts[0][i] * hidden_acts[i] * (1-hidden_acts[i]))
	"""
	print(output_wgts[1])
	#print(hidden_acts)
	for i in range(num_hidden_nodes):
		weighted_sum = 0
		for j in range(num_output_nodes):
			for k in range(len(output_wgts[j])):
				print(output_wgts[j][k])
				weighted_sum += output_wgts[j][k]*delta_o[j]
			#weighted_sum = np.dot(hidden_wgts[i], delta_o[j])
		print(weighted_sum)
		delta_h.append(-1*(hidden_acts[i] * (1 - hidden_acts[i]) * weighted_sum))
	print(delta_h)
	"""
	#print(delta_h)
	return delta_h

# Update output layer weights
def update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, eta):
	new_output_wgts = deepcopy(output_wgts)
	new_hidden_acts = deepcopy(hidden_acts)
	new_hidden_acts.append(1)
	print(new_hidden_acts)
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

input_set = [[1,0],[0,0],[0,1],[1,1]]
targets = [0,1,0,1]
num_hidden_nodes = 2
num_output_nodes = 1
#hidden_wgts = [[0.1 for _ in range(len(input_set[0])+1)] for _ in range(num_hidden_nodes)]
hidden_wgts = [[2,-2,0],[1,3,-1]]
output_wgts = [[3,-2,-1]]
#output_wgts = [[0.1 for _ in range(num_hidden_nodes)] for _ in range(num_output_nodes)]
eta = 1
#for _ in range(10000):
	#for m in range(4):
m = 0

bias = 1
for i in range(num_hidden_nodes):
	input_set[i].append(bias)

hidden_acts = hidden_act(input_set, hidden_wgts, num_hidden_nodes, m)
output_acts = output_act(hidden_acts, output_wgts, num_hidden_nodes,num_output_nodes)
delta_o = compute_output_error(output_acts, num_output_nodes, targets[m])
delta_h = compute_hidden_error(hidden_acts, output_wgts, num_hidden_nodes, num_output_nodes, 
	delta_o)
output_wgts = update_output_wgts(hidden_acts, num_hidden_nodes, num_output_nodes, output_wgts, delta_o, 
	eta)
hidden_wgts = update_hidden_wgts(input_set, num_hidden_nodes, hidden_wgts, delta_h, eta, m)

#for m in range(4):
	#m = 2
#print(input_set[m])
#hidden_acts = hidden_act(input_set, hidden_wgts, num_hidden_nodes, m)
#print(hidden_acts)
#output_acts = output_act(hidden_acts, output_wgts, num_hidden_nodes,num_output_nodes)
#print(output_acts)
#print(compute_output_error(output_acts, num_output_nodes, target_wgts))
#delta_o = compute_output_error(output_acts, num_output_nodes, targets[m])
#print(delta_o)
#print()

