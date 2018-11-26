from Helpers import Image, MultiLayerPerceptron, AllWeightsValid
from random import randint, shuffle

def Problem3(x_train, y_train, p_width, p_height, greyscale_range):
	# Declare training and test sets for digits 0-9
	all_train = [[] for _ in range(10)]
	all_test = [[] for _ in range(10)]

	# Step 1: Divide each 28x28 image into a 7x7 grid of 4x4 squares
	# Get the average grayscale value of each square. This average will
	# be used as a feature
	# x_train and x_test are uint8 arrays of grayscale image data with
	# shape (num_samples, 28, 28).
	# y_train and y_test are uint8 arrays of digit labels 
	# (integers in range 0-9) with shape (num_samples,).
	img_4x4_train_set = [0 for _ in range(49)]
	for i in range(len(x_train)):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_train[0][x][y]
				img_4x4_train_set[j*7//4+k//4] = total

	"""img_4x4_test_set = [[0 for _ in range(7)] for _ in range(7)]
	for i in range(len(x_test)):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_train[0][x][y]
				img_4x4_test_set[j//4][k//4] = total"""

	# Step 2: Select the number of neurons in the hidden layer
	# n = 3, 6, 10, 15, 21
	# There will be ten output neurons.
	# Perform 1000 iterations for each value of n
	p_3  = MultiLayerPerceptron(img_4x4_train_set,  3, 10, 10, y_train[0])
	#p_6  = MultiLayerPerceptron(img_4x4_train_set,  6, 10, 1000, y_train[0])
	#p_10 = MultiLayerPerceptron(img_4x4_train_set, 10, 10, 1000, y_train[0])
	#p_15 = MultiLayerPerceptron(img_4x4_train_set, 15, 10, 1000, y_train[0])
	#p_21 = MultiLayerPerceptron(img_4x4_train_set, 21, 10, 1000, y_train[0])
	print(p_3.GetOutputError())

	# Step 3: After each iteration, test performance will be used to determine
	# its error rate.

	# Step 4: Select the value of n with the minimum error rate.

	# Step 5: Use the best performing neural network to identify the digit in a
	# given test image