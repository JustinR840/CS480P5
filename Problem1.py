from Helpers import Perceptron
from random import randint, shuffle
import numpy as np


def DoTesting(p, orig_train7, orig_train9):
	confusion_matrix = np.zeros((2, 2))
	# Predicted is the row
	# Actual is the column
	# [[ n_9_right, n_9_wrong ],
	#  [ n_7_wrong, n_7_right ]]

	test = np.concatenate((orig_train7, orig_train9))
	test_targets = np.concatenate((np.ones(len(orig_train7), dtype=int), np.zeros(len(orig_train9), dtype=int)))

	test_indices = np.arange(0, len(test))
	shuffle(test_indices)


	for i in test_indices:
		activation = p.ActivationValue(test[i])

		confusion_matrix[activation][test_targets[i]] += 1

	err = (confusion_matrix[0][1] + confusion_matrix[1][0]) / len(test)

	return err


def DoTraining(p, n_epochs, trainindicies, train, traintargets, test7, test9):
	lowest_err = DoTesting(p,test7, test9)
	bestweights = np.array(list(p.weights))

	n_epochs_passed = 0
	while (n_epochs_passed < n_epochs):
		if (n_epochs_passed % 100 == 0):
			print("Trained perceptron on", n_epochs_passed, "epochs.")

		for i in trainindicies:
			activation = p.ActivationValue(train[i])

			p.UpdateWeights(train[i], activation, traintargets[i])

		current_err = DoTesting(p, test7, test9)
		if(current_err < lowest_err):
			bestweights = np.array(list(p.weights))

		n_epochs_passed += 1

	p.weights = np.array(list(bestweights))


def Problem1(orig_train7, orig_train9, orig_test7, orig_test9):
	#train7 = orig_train7[0:round(len(orig_train7) * 0.8)]
	#train9 = orig_train9[0:round(len(orig_train9) * 0.8)]

	train = np.concatenate((orig_train7, orig_train9))
	traintargets = np.concatenate((np.ones(len(orig_train7)), np.zeros(len(orig_train9))))


	trainindicies = np.arange(0, len(train))
	shuffle(trainindicies)

	weights = [randint(-10, 11) * 0.01 for _ in range(len(train[0]))]
	eta = 0.1
	n_epochs = 1000

	p = Perceptron(weights, eta)

	#test7 = np.array(orig_train7[0:round(len(orig_train7) * 0.2)])
	#test9 = np.array(orig_train9[0:round(len(orig_train9) * 0.2)])

	DoTraining(p, n_epochs, trainindicies, train, traintargets, orig_test7, orig_test9)

	np.set_printoptions(linewidth=200)
	print("Best Weights: " + str(p.weights))

	err = DoTesting(p, orig_train7, orig_train9)
	print("Error: " + str(err))
