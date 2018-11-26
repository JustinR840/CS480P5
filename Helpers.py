import numpy as np
import math
import random as r

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


def AllWeightsValid(perceptron, inputs, targets):
	for i in range(len(inputs)):
		if(perceptron.ActivationValue(inputs[i]) != targets[i]):
			return False
	return True


class MulticlassPerceptron:
	def __init__(self, weights, eta):
		self.weights = weights
		self.eta = eta

	def ActivationLabel(self, input):
		maxsum = 0
		label = 0

		for i in range(len(self.weights)):
			sum = 0.0

			for j in range(len(input)):
				sum += self.weights[i][j] * input[j]

			if(sum > maxsum):
				maxsum = sum
				label = i

		return label


	def UpdateWeights(self, input, predicted_label, target_label):
		for i in range(len(self.weights[predicted_label])):
			self.weights[predicted_label][i] = self.weights[predicted_label][i] - self.eta * input[i]

		for i in range(len(self.weights[target_label])):
			self.weights[target_label][i] = self.weights[target_label][i] + self.eta * input[i]


class Perceptron:
	def __init__(self, weights, eta):
		self.weights = weights
		self.eta = eta

	def ActivationValue(self, input):
		sum = 0.0

		for i in range(len(input)):
			sum += self.weights[i] * input[i]

		if(sum > 0):
			return 1
		return 0

	def UpdateWeights(self, input, activation, target):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - self.eta * (activation - target) * input[i]


class Image:
	def __init__(self, image, number, pixelWidth, pixelHeight, greyscaleRange):
		# image should be any type of container than is 2 dimensional and can be iterated over using indexing
		self.__image = image
		self.__number = number
		self.__pixelWidth = pixelWidth
		self.__pixelHeight = pixelHeight
		self.__greyscaleRange = greyscaleRange

		self.__density = -1
		self.__degreeOfSymmetry = -1
		self.__maxVerticalIntersections = -1
		self.__avgVerticalIntersections = -1
		self.__maxHorizontalIntersections = -1
		self.__avgHorizontalIntersections = -1

		self.__SetDensity()
		self.__SetDegreeOfSymmetry()
		self.__SetAvgAndMaxVerticalIntersections()
		self.__SetAvgAndMaxHorizontalIntersections()

	# 1: Density: Density is defined as the average gray scale value of all the pixels in the image
	# and is thus a real number between 0 and 255.
	def __SetDensity(self):
		total = 0

		for row in range(len(self.__image)):
			for col in range(len(self.__image[row])):
				total += self.__image[row][col]

		self.__density = total / (self.__pixelHeight * self.__pixelWidth)

	# 2: Measure of symmetry is defined as the average gray scale of the image obtained by the
	# bitwise XOR (⊕) of each pixel with its corresponding vertically reflected image.
	# (Thus if I is the image, let I' be the image whose j-th column is the (28 − j)-th column of I. Then,
	# the measure of symmetry is the density of I ⊕ I'.)
	def __SetDegreeOfSymmetry(self):
		total = 0

		for row in range(self.__pixelHeight):
			for col in range(self.__pixelWidth):
				normal_val = self.__image[row][col]
				inverted_val = self.__image[row][self.__pixelWidth - col - 1]

				# XOR the original position with the flipped position
				total += normal_val ^ inverted_val

		# Recall that this is the density of the XORed image
		density = total / (self.__pixelWidth * self.__pixelHeight)

		self.__degreeOfSymmetry = density

	# 3: The number of vertical intersections is defined as follows:
	# First turn the image into black and white image by assigning a color 0 (1) for gray scale values above (below) 128.
	# Consider the j-th column as a string of 0’s and 1’s and count the number of changes from 0 to 1 or 1 to 0.
	# For example, the number of changes for string 11011001 is 4.
	# Average this over all columns to get the average number of vertical intersections, and maximum over all columns will
	# give the maximum number of vertical intersections.
	# Vertical measures are defined in a similar way based on rows.
	def __SetAvgAndMaxVerticalIntersections(self):
		# Check for 1->0 and 0->1 swaps of each pixel in the image and count them
		# This loop goes over the image using COLUMN MAJOR ORDER
		column_swaps = [0 for _ in range(self.__pixelWidth)]
		for col in range(self.__pixelWidth):
			last_val = 0 if self.__image[0][col] >= self.__greyscaleRange // 2 else 1

			for row in range(1, self.__pixelHeight):
				current_val = 0 if self.__image[row][col] >= self.__greyscaleRange // 2 else 1
				if(current_val != last_val):
					last_val = current_val
					column_swaps[col] += 1

		# Calculate the average vertical intersections and max vertical intersections
		total = 0
		for i in range(len(column_swaps)):
			total += column_swaps[i]
		avg = total / len(column_swaps)

		self.__maxVerticalIntersections = max(column_swaps)
		self.__avgVerticalIntersections = avg

	def __SetAvgAndMaxHorizontalIntersections(self):
		# Check for 1->0 and 0->1 swaps of each pixel in the image and count them
		# This loop goes over the image using ROW MAJOR ORDER
		row_swaps = [0 for _ in range(self.__pixelHeight)]
		for row in range(self.__pixelHeight):
			last_val = 0 if self.__image[row][0] >= self.__greyscaleRange // 2 else 1

			for col in range(1, self.__pixelWidth):
				current_val = 0 if self.__image[row][col] >= self.__greyscaleRange // 2 else 1
				if(current_val != last_val):
					last_val = current_val
					row_swaps[row] += 1

		# Calculate the average horizontal intersections and max horizontal intersections
		total = 0
		for i in range(len(row_swaps)):
			total += row_swaps[i]
		avg = total / len(row_swaps)

		self.__maxHorizontalIntersections = max(row_swaps)
		self.__avgHorizontalIntersections = avg

	def GetImage(self):
		return self.__image

	def GetNumber(self):
		return self.__number

	def GetPixelWidth(self):
		return self.__pixelWidth

	def GetPixelHeight(self):
		return self.__pixelHeight

	def GetGreyscaleRange(self):
		return self.__greyscaleRange

	def GetDensity(self):
		return self.__density

	def GetDegreeOfSymmetry(self):
		return self.__degreeOfSymmetry

	def GetAvgVerticalIntersections(self):
		return self.__avgVerticalIntersections

	def GetMaxVerticalIntersections(self):
		return self.__maxVerticalIntersections

	def GetAvgHorizontalIntersections(self):
		return self.__avgHorizontalIntersections

	def GetMaxHorizontalIntersections(self):
		return self.__maxHorizontalIntersections


#def MultiLayerPerceptron(input, num_hidden_layers, num_output_nodes, 
	#num_epochs):

class MultiLayerPerceptron:
	def __init__(self, input_set, num_hidden_nodes, 
		num_output_nodes, num_epochs, target):

		# Initialize the random number generator
		#random.seed()

		# Initialize the hidden layer vector weights
		self.hidden_wgts = [[0.1 for _ in range(49)]
		for _ in range(num_hidden_nodes)]

		# Initialize the output vector weights
		self.output_wgts = [[0.1 for _ in range(num_hidden_nodes)]
		for _ in range(num_output_nodes)]

		# Initialize the target vector weights
		self.targets = [0 for _ in range(num_output_nodes)]
		self.targets[target] = 1


		# Train the data set for a number of epochs
		#for _ in range(num_epochs):
		for _ in range(1):

			# Training
			# Forwards phase
			# Compute activation of hidden neurons
			self.hidden_acts = []
			for i in range(num_hidden_nodes):
				self.hidden_acts.append(1 / (1 + math.exp(-1 * 
				np.dot(np.array(input_set), np.array(self.hidden_wgts[i])))))

			# Compute activation of output neurons
			self.output_acts = []
			for i in range(num_output_nodes):
				self.output_acts.append(1 / (1 + math.exp(-1 * 
				np.dot(np.array(self.hidden_acts),
				np.array(self.output_wgts[i])))))

			# Backwards phase
			# Compute error at the output neurons
			self.delta_o = []
			for i in range(num_output_nodes):
				self.delta_o.append((self.output_acts[i] - self.targets[i])
				* self.output_acts[i] * ((1 - self.output_acts[i])))

			# Compute error at the hidden layer neurons
			self.delta_h = []
			for i in range(num_hidden_nodes):
				self.delta_h.append(self.hidden_acts[i] * (1 - self.hidden_acts[i]) 
				* (np.dot(np.array(self.hidden_wgts[i]),np.array(self.delta_o[i]))))

			self.eta = 0.01

			# Update output layer weights
			for i in range(num_output_nodes):
				for j in range(num_hidden_nodes):
					self.output_wgts[i][j] = (self.output_wgts[i][j] - self.eta
					*self.delta_o[i] * self.hidden_acts[j])

			# Update hidden layer weights
			for i in range(num_hidden_nodes):
				for j in range(len(input_set)):
					self.hidden_wgts[i][j] = (self.hidden_wgts[i][j] - self.eta
					* self.delta_h[i] * input_set[j])

	def GetOutputError(self):
		return self.delta_o