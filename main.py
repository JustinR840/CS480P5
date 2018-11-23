import tensorflow as tf
import numpy as np
from Helpers import Image, Perceptron, AllWeightsValid
from random import randint



def main():
	p_width = 28
	p_height = 28
	greyscale_range = 255

	# Just to verify that the internals of Image still work as expected
	TestSimpleData(p_width, p_height, greyscale_range)





	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	train1 = []
	train5 = []

	test7 = []
	test9 = []


	for i in range(len(y_test)):
		if (y_test[i] == 1):
			train1.append(Image(x_test[i], 1, p_width, p_height, greyscale_range))
		elif(y_test[i] == 5):
			train5.append(Image(x_test[i], 5, p_width, p_height, greyscale_range))
		elif(y_test[i] == 7):
			test7.append(Image(x_test[i], 7, p_width, p_height, greyscale_range))
		elif(y_test[i] == 9):
			test9.append(Image(x_test[i], 9, p_width, p_height, greyscale_range))



	inputs = [
		[0, 0, -1],
		[0, 1, -1],
		[1, 0, -1],
		[1, 1, -1]
	]

	targets = [0, 0, 0, 1]

	eta = 0.1
	n_epochs = 1000

	weights = [randint(-10, 11) * 0.01 for _ in range(len(inputs[0]))]

	p = Perceptron(weights, eta)


	while(not AllWeightsValid(p, inputs, targets)):
		for i in range(len(inputs)):
			activation = p.ActivationValue(inputs[i])

			p.UpdateWeights(inputs[i], activation, targets[i])








def TestSimpleData(p_width, p_height, greyscale_range):
	# [0  , 0  ]
	# [255, 255]
	# The following data should have:
	#	degreeOfSymmetry = 0
	#	maxVerticalIntersections = 1
	#	avgVerticalIntersections = 1
	#	maxHorizontalIntersections = 0
	#	avgHorizontalIntersections = 0
	data1 = [[0 if y < (p_height // 2) else greyscale_range for _ in range(p_width)] for y in range(p_height)]

	# [0, 255]
	# [0, 255]
	# The following data should have:
	#	degreeOfSymmetry = 255
	#	maxVerticalIntersections = 0
	#	avgVerticalIntersections = 0
	#	maxHorizontalIntersections = 1
	#	avgHorizontalIntersections = 1
	data2 = [[0 if x < (p_width // 2) else greyscale_range for x in range(p_width)] for _ in range(p_height)]


	img1 = Image(data1, 0, p_width, p_height, greyscale_range)
	img2 = Image(data2, 0, p_width, p_height, greyscale_range)


	assert(img1.GetDegreeOfSymmetry() == 0)
	assert(img1.GetMaxVerticalIntersections() == 1)
	assert(img1.GetAvgVerticalIntersections() == 1)
	assert(img1.GetMaxHorizontalIntersections() == 0)
	assert(img1.GetAvgHorizontalIntersections() == 0)

	assert(img2.GetDegreeOfSymmetry() == greyscale_range)
	assert(img2.GetMaxVerticalIntersections() == 0)
	assert(img2.GetAvgVerticalIntersections() == 0)
	assert(img2.GetMaxHorizontalIntersections() == 1)
	assert(img2.GetAvgHorizontalIntersections() == 1)










main()