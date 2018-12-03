from Helpers import Image, Perceptron, AllWeightsValid
from random import randint, shuffle
import numpy as np


def DoTesting(p, orig_train7, orig_train9):
	n_7_right = 0
	n_7_wrong = 0
	n_9_right = 0
	n_9_wrong = 0

	test7 = orig_train7[round(len(orig_train7) * 0.8):len(orig_train7)]
	test9 = orig_train9[round(len(orig_train9) * 0.8):len(orig_train9)]

	test = np.concatenate((test7, test9))
	testtargets = np.concatenate((np.ones(len(test7)), np.zeros(len(test9))))

	testindicies = np.arange(0, len(test))
	shuffle(testindicies)

	for i in testindicies:
		activation = p.ActivationValue(test[i])

		if (testtargets[i] == 1):
			if (activation == 1):
				n_7_right += 1
			else:
				n_7_wrong += 1
		else:
			if (activation == 0):
				n_9_right += 1
			else:
				n_9_wrong += 1

	return ((n_7_right / (n_7_right + n_7_wrong)) + (n_9_right / (n_9_right + n_9_wrong))) / 2, (n_7_right / (n_7_right + n_7_wrong)), (n_9_right / (n_9_right + n_9_wrong))


def Problem1B(orig_train7, orig_train9):
	train7 = orig_train7[0:round(len(orig_train7) * 0.8)]
	train9 = orig_train9[0:round(len(orig_train9) * 0.8)]

	train = np.concatenate((train7, train9))
	traintargets = np.concatenate((np.ones(len(train7)), np.zeros(len(train9)))) # [1 for _ in range(len(train7))] + [0 for _ in range(len(train9))]


	trainindicies = np.arange(0, len(train))
	shuffle(trainindicies)

	weights = [randint(-10, 11) * 0.01 for _ in range(len(train[0]))]
	eta = 0.1
	n_epochs = 1000

	p = Perceptron(weights, eta)

	besttot, besta, bestb = DoTesting(p, orig_train7[0:round(len(orig_train7) * 0.2)], orig_train9[0:round(len(orig_train9) * 0.2)])
	bestweights = np.array(list(weights))

	stop = 0
	while (stop < n_epochs):
		if (stop % 100 == 0):
			print("Gone through", stop, "epochs on the TRAIN set")

		for i in trainindicies:
			activation = p.ActivationValue(train[i])

			p.UpdateWeights(train[i], activation, traintargets[i])

		currenttot, currenta, currentb = DoTesting(p, orig_train7[0:round(len(orig_train7) * 0.2)], orig_train9[0:round(len(orig_train9) * 0.2)])
		if(currenttot > besttot): # and (currenta > besta or currentb > bestb)):
			bestweights = np.array(list(p.weights))

		stop += 1

	print(p.weights)
	p.weights = np.array(list(bestweights))
	print(p.weights)

	print(DoTesting(p, orig_train7, orig_train9))





def Problem1(x_train, y_train, p_width, p_height, greyscale_range):
	train7 = []
	train9 = []

	for i in range(len(y_train)):
		if (i % 100 == 0):
			print("Transformed", i, "entries into Images")

		if (y_train[i] == 7):
			train7.append(Image(x_train[i], 7, p_width, p_height, greyscale_range))
		elif (y_train[i] == 9):
			train9.append(Image(x_train[i], 9, p_width, p_height, greyscale_range))

	train7 = train7[:round(len(train7) * 0.8)]
	train9 = train9[:round(len(train9) * 0.8)]
	test7 = train7[:round(len(train7) * -0.2)]
	test9 = train9[:round(len(train9) * -0.2)]

	train = train7 + train9
	shuffle(train)

	train_inputs = []
	train_targets = []
	for i in range(len(train)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TRAIN set")

		m_input = []
		m_input.append(train[i].GetDensity())
		m_input.append(train[i].GetDegreeOfSymmetry())
		m_input.append(train[i].GetAvgVerticalIntersections())
		m_input.append(train[i].GetMaxVerticalIntersections())
		m_input.append(train[i].GetAvgHorizontalIntersections())
		m_input.append(train[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		train_inputs.append(m_input)

		if (train[i].GetNumber() == 7):
			train_targets.append(1)
		else:
			train_targets.append(0)

	weights = [randint(-10, 11) * 0.01 for _ in range(len(train_inputs[0]))]
	eta = 0.1
	n_epochs = 1000

	p = Perceptron(weights, eta)

	stop = 0
	while (stop < n_epochs and not AllWeightsValid(p, train_inputs, train_targets)):
		if (stop % 100 == 0):
			print("Gone through", stop, "epochs on the TRAIN set")
		for i in range(len(train_inputs)):
			activation = p.ActivationValue(train_inputs[i])

			#p.UpdateWeights(np.array(train_inputs[i]), activation, train_targets[i])
			p.UpdateWeights(train_inputs[i], activation, train_targets[i])

		stop += 1

	print(p.weights)

	# m_weights = [-10.279285714275588, 6.928367346936257, 59.25285714287503, -87.7399999998952, -550.8614285721433, 21.159999999977785, -1255.4899999999539]
	# eta = 0.1
	# p = Perceptron(m_weights, eta)


	test = test7 + test9
	shuffle(test)

	test_inputs = []
	test_targets = []
	for i in range(len(test)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TEST set")

		m_input = []
		m_input.append(test[i].GetDensity())
		m_input.append(test[i].GetDegreeOfSymmetry())
		m_input.append(test[i].GetAvgVerticalIntersections())
		m_input.append(test[i].GetMaxVerticalIntersections())
		m_input.append(test[i].GetAvgHorizontalIntersections())
		m_input.append(test[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		test_inputs.append(m_input)

		if (test[i].GetNumber() == 7):
			test_targets.append(1)
		else:
			test_targets.append(0)

	n_7_right = 0
	n_7_wrong = 0
	n_9_right = 0
	n_9_wrong = 0

	for i in range(len(test_inputs)):
		if (i % 100 == 0):
			print("Gotten results for ", i, "entries in the TEST set")

		activation = p.ActivationValue(test_inputs[i])

		if (test_targets[i] == 1):
			if (activation == 1):
				n_7_right += 1
			else:
				n_7_wrong += 1
		else:
			if (activation == 0):
				n_9_right += 1
			else:
				n_9_wrong += 1

	print("Percentage of 7s correctly guessed: " + str((n_7_right / (n_7_right + n_7_wrong)) * 100) + "%")
	print("Percentage of 9s correctly guessed: " + str((n_9_right / (n_9_right + n_9_wrong)) * 100) + "%")
