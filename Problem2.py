from Helpers import Image, MulticlassPerceptron
from random import randint, shuffle
import numpy as np


def DataToImages(x_train, y_train, p_width, p_height, greyscale_range, all_train):
	for i in range(len(y_train)):
		if (i % 100 == 0):
			print("Transformed", i, "entries into Images")

		all_train[int(y_train[i])].append(Image(x_train[i], y_train[i], p_width, p_height, greyscale_range))

def ImagesToInput(training_data, train_inputs, train_targets):
	for i in range(len(training_data)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TRAIN set")

		m_input = []
		m_input.append(training_data[i].GetDensity())
		m_input.append(training_data[i].GetDegreeOfSymmetry())
		m_input.append(training_data[i].GetAvgVerticalIntersections())
		m_input.append(training_data[i].GetMaxVerticalIntersections())
		m_input.append(training_data[i].GetAvgHorizontalIntersections())
		m_input.append(training_data[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		train_inputs.append(m_input)

		train_targets.append(training_data[i].GetNumber())

def DoTraining(train_inputs, train_targets):
	weights = np.random.uniform(-0.1, 0.1, [10, len(train_inputs[0])])
	eta = 0.1
	n_epochs = 1000

	p = MulticlassPerceptron(weights, eta)

	stop = 0
	while (stop < n_epochs):# and not AllWeightsValid(p, train_inputs, train_targets)):
		if (stop % 100 == 0):
			print("Gone through", stop, "epochs on the TRAIN set")

		for i in range(len(train_inputs)):
			activation = p.ActivationLabel(train_inputs[i])

			if(activation != train_targets[i]):
				p.UpdateWeights(train_inputs[i], activation, train_targets[i])

		stop += 1

	return p

def TransformTestingData(test_data, test_inputs, test_targets):
	for i in range(len(test_data)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TEST set")

		m_input = []
		m_input.append(test_data[i].GetDensity())
		m_input.append(test_data[i].GetDegreeOfSymmetry())
		m_input.append(test_data[i].GetAvgVerticalIntersections())
		m_input.append(test_data[i].GetMaxVerticalIntersections())
		m_input.append(test_data[i].GetAvgHorizontalIntersections())
		m_input.append(test_data[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		test_inputs.append(m_input)

		test_targets.append(test_data[i].GetNumber())


def DoTesting(p, test_inputs, test_targets):
	confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]

	for i in range(len(test_inputs)):
		if (i % 100 == 0):
			print("Gotten results for ", i, "entries in the TEST set")

		activation = p.ActivationLabel(test_inputs[i])

		confusion_matrix[activation][test_targets[i]] += 1

	print(np.array(confusion_matrix))

def Problem2(x_train, y_train, p_width, p_height, greyscale_range):
	all_train = [[] for _ in range(10)]
	all_test = [[] for _ in range(10)]
	DataToImages(x_train, y_train, p_width, p_height, greyscale_range, all_train)

	for i in range(len(all_train)):
		all_train[i] = all_train[i][:round(len(all_train[i]) * 0.8)]
		all_test[i] = all_train[i][:round(len(all_train[i]) * -0.2)]


	training_data = []
	for i in range(len(all_train)):
		training_data = training_data + all_train[i]
	shuffle(training_data)

	train_inputs = []
	train_targets = []
	ImagesToInput(training_data, train_inputs, train_targets)

	return

	p = DoTraining(train_inputs, train_targets)
	print(p.weights)


	test_data = []
	for i in range(len(all_test)):
		test_data = test_data + all_test[i]
	shuffle(test_data)

	test_inputs = []
	test_targets = []
	TransformTestingData(test_data, test_inputs, test_targets)
	DoTesting(p, test_inputs, test_targets)

