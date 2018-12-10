from Helpers import MulticlassPerceptron
from random import randint, shuffle
import numpy as np


def DoTraining(train_inputs, train_targets, train_indices, test_input, test_targets, test_indices):
	weights = np.random.uniform(-0.1, 0.1, [10, len(train_inputs[0])])
	eta = 0.1
	n_epochs = 1000

	p = MulticlassPerceptron(weights, eta)

	lowest_err, c_m = DoTesting(p, test_input, test_targets, test_indices)
	bestweights = np.array(list(p.weights))

	stop = 0
	while (stop < n_epochs):
		if (stop % 100 == 0):
			print("Gone through", stop, "epochs on the TRAIN set")

		for i in train_indices:
			activation = p.ActivationLabel(train_inputs[i])

			if(activation != train_targets[i]):
				p.UpdateWeights(train_inputs[i], activation, train_targets[i])

		current_err, c_m = DoTesting(p, test_input, test_targets, test_indices)
		if(current_err < lowest_err):
			bestweights = np.array(list(p.weights))

		stop += 1

	p.weights = np.array(list(bestweights))

	return p


def DoTesting(p, test_inputs, test_targets, test_indices):
	confusion_matrix = np.zeros((10, 10))

	for i in test_indices:
		activation = p.ActivationLabel(test_inputs[i])

		confusion_matrix[activation][test_targets[i].astype(int)] += 1

	total = 0.0
	for i in range(len(confusion_matrix)):
		for j in range(len(confusion_matrix)):
			if(i != j):
				total += confusion_matrix[i][j]

	err = total / (10 * 10)

	return err, confusion_matrix

def Problem2(train_sets, test_sets):
	all_train = np.concatenate((train_sets[0], train_sets[1], train_sets[2], train_sets[3], train_sets[4], train_sets[5], train_sets[6], train_sets[7], train_sets[8], train_sets[9]))
	train_targets = np.concatenate((np.zeros(len(train_sets[0])), np.ones(len(train_sets[1])), np.full(len(train_sets[2]), 2), np.full(len(train_sets[3]), 3), np.full(len(train_sets[4]), 4), np.full(len(train_sets[5]), 5), np.full(len(train_sets[6]), 6), np.full(len(train_sets[7]), 7), np.full(len(train_sets[8]), 8), np.full(len(train_sets[9]), 9)))

	all_test = np.concatenate((test_sets[0], test_sets[1], test_sets[2], test_sets[3], test_sets[4], test_sets[5], test_sets[6], test_sets[7], test_sets[8], test_sets[9]))
	test_targets = np.concatenate((np.zeros(len(test_sets[0])), np.ones(len(test_sets[1])), np.full(len(test_sets[2]), 2), np.full(len(test_sets[3]), 3), np.full(len(test_sets[4]), 4), np.full(len(test_sets[5]), 5), np.full(len(test_sets[6]), 6), np.full(len(test_sets[7]), 7), np.full(len(test_sets[8]), 8), np.full(len(test_sets[9]), 9)))

	train_indices = np.arange(0, len(all_train))
	shuffle(train_indices)

	test_indices = np.arange(0, len(all_test))
	shuffle(test_indices)

	p = DoTraining(all_train, train_targets, train_indices, all_test, test_targets, test_indices)

	np.set_printoptions(linewidth=200)
	print("Best Weights: " + str(p.weights))

	err, c_m = DoTesting(p, all_test, test_targets, test_indices)
	print("Error: " + str(err))
	print("Confusion Matrix: " + str(c_m))

	return p