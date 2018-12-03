import numpy as np
import math
import random as r
from copy import deepcopy
import json

def GetDegreeOfSymmetry(image, pixelHeight, pixelWidth):
	total = 0
	for row in range(pixelHeight):
		for col in range(pixelWidth):
			normal_val = image[row][col]
			inverted_val = image[row][pixelWidth - col - 1]

			# XOR the original position with the flipped position
			total += normal_val ^ inverted_val

	# Recall that this is the density of the XORed image
	return total / (pixelWidth * pixelHeight)


def GetAvgAndMaxVerticalIntersections(image, pixelHeight, pixelWidth):
	# Check for 1->0 and 0->1 swaps of each pixel in the image and count them
	# This loop goes over the image using COLUMN MAJOR ORDER
	column_swaps = np.zeros(pixelWidth)
	for col in range(pixelWidth):
		last_val = image[0][col]

		for row in range(1, pixelHeight):
			current_val = image[row][col]
			if (current_val != last_val):
				last_val = current_val
				column_swaps[col] += 1

	# Calculate the average vertical intersections and max vertical intersections
	total = 0
	for i in range(len(column_swaps)):
		total += column_swaps[i]
	avg = total / len(column_swaps)

	return max(column_swaps), avg


def GetAvgAndMaxHorizontalIntersections(image, pixelHeight, pixelWidth):
	# Check for 1->0 and 0->1 swaps of each pixel in the image and count them
	# This loop goes over the image using ROW MAJOR ORDER
	row_swaps = np.zeros(pixelHeight)
	for row in range(pixelHeight):
		last_val = image[row][0]

		for col in range(1, pixelWidth):
			current_val = image[row][col]
			if (current_val != last_val):
				last_val = current_val
				row_swaps[row] += 1

	# Calculate the average horizontal intersections and max horizontal intersections
	total = 0
	for i in range(len(row_swaps)):
		total += row_swaps[i]
	avg = total / len(row_swaps)

	return max(row_swaps), avg


def LoadInputs(x_set, y_set):
	counts = [0 for _ in range(10)]
	for i in y_set:
		counts[i] += 1

	input_set = np.array([[[0, 0, 0, 0, 0, 0, 0] for _ in range(counts[i])] for i in range(10)])

	for i in range(len(x_set)):
		if (i % 1000 == 0):
			print("Transformed", i, "images into input.")

		image = x_set[i]
		number = y_set[i]
		pixelWidth = 28
		pixelHeight = 28

		density = np.average(np.ndarray.flatten(image))
		degreeOfSymmetry = GetDegreeOfSymmetry(image, pixelHeight, pixelWidth)
		blackwhite = image <= 128
		maxVerticalIntersections, avgVerticalIntersections = GetAvgAndMaxVerticalIntersections(blackwhite, pixelHeight, pixelWidth)
		maxHorizontalIntersections, avgHorizontalIntersections = GetAvgAndMaxHorizontalIntersections(blackwhite, pixelHeight, pixelWidth)

		input_set[number][counts[number] - 1] = [density, degreeOfSymmetry, avgVerticalIntersections, maxVerticalIntersections, avgHorizontalIntersections, maxHorizontalIntersections, -1]

		counts[number] -= 1



	return input_set



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

	image1 = np.array(data1)
	density1 = np.average(np.ndarray.flatten(image1))
	degreeOfSymmetry1 = GetDegreeOfSymmetry(image1, p_height, p_width)
	blackwhite1 = image1 <= 128
	maxVerticalIntersections1, avgVerticalIntersections1 = GetAvgAndMaxVerticalIntersections(blackwhite1, p_height, p_width)
	maxHorizontalIntersections1, avgHorizontalIntersections1 = GetAvgAndMaxHorizontalIntersections(blackwhite1, p_height, p_width)

	img1 = [density1, degreeOfSymmetry1, avgVerticalIntersections1, maxVerticalIntersections1, avgHorizontalIntersections1, maxHorizontalIntersections1, -1]

	image2 = np.array(data2)
	density2 = np.average(np.ndarray.flatten(image2))
	degreeOfSymmetry2 = GetDegreeOfSymmetry(image2, p_height, p_width)
	blackwhite2 = image2 <= 128
	maxVerticalIntersections2, avgVerticalIntersections2 = GetAvgAndMaxVerticalIntersections(blackwhite2, p_height, p_width)
	maxHorizontalIntersections2, avgHorizontalIntersections2 = GetAvgAndMaxHorizontalIntersections(blackwhite2, p_height, p_width)

	img2 = [density2, degreeOfSymmetry2, avgVerticalIntersections2, maxVerticalIntersections2, avgHorizontalIntersections2, maxHorizontalIntersections2, -1]




	assert(img1[1] == 0)
	assert(img1[2] == 1)
	assert(img1[3] == 1)
	assert(img1[4] == 0)
	assert(img1[5] == 0)

	assert(img2[1] == greyscale_range)
	assert(img2[2] == 0)
	assert(img2[3] == 0)
	assert(img2[4] == 1)
	assert(img2[5] == 1)


class MulticlassPerceptron:
	# Relies on NumPy arrays

	def __init__(self, weights, eta):
		self.weights = weights
		self.eta = eta

	def ActivationLabel(self, input):
		results = np.dot(self.weights, input)
		return np.argmax(results)

	def UpdateWeights(self, input, predicted_label, target_label):
		self.weights[target_label.astype(int)] = self.weights[target_label.astype(int)] + self.eta * input
		self.weights[predicted_label.astype(int)] = self.weights[predicted_label.astype(int)] - self.eta * input


class Perceptron:
	# Relies on NumPy arrays

	def __init__(self, weights, eta):
		self.weights = weights
		self.eta = eta

	def ActivationValue(self, input):
		sum = np.dot(self.weights, input)

		if(sum > 0):
			return 1
		return 0

	def UpdateWeights(self, input, activation, target):
		self.weights = self.weights - (self.eta * (activation - target)) * np.array(input)


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

		#self.__booleanTransitionCalculator = lambda t: 1 if t >= self.__greyscaleRange // 2 else -1

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

		# booleans = np.array([[self.__booleanTransitionCalculator(j) for j in i] for i in self.__image])
		# booleanchanges = np.sum((np.diff(booleans) != 0) * 1, 1)
		#
		# self.__maxVerticalIntersections = np.amax(booleanchanges)
		# self.__avgVerticalIntersections = np.average(booleanchanges)
		#
		# return


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

		# np.set_printoptions(linewidth=200)

		# print(booleans)
		# print(A)
		# print(booleanchanges)
		#
		# print(column_swaps)
		# print(self.__image)



		self.__maxVerticalIntersections = max(column_swaps)
		self.__avgVerticalIntersections = avg

	def __SetAvgAndMaxHorizontalIntersections(self):

		# booleans = np.array([[self.__booleanTransitionCalculator(j) for j in i] for i in self.__image.T])
		# booleanchanges = np.sum((np.diff(booleans) != 0) * 1, 1)
		#
		# self.__maxHorizontalIntersections = np.amax(booleanchanges)
		# self.__avgHorizontalIntersections = np.average(booleanchanges)
		#
		# return


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

####################################################################################################
############################ Multilayer perceptron helper functions ################################
####################################################################################################

####################################################################################################
############################ Multilayer perceptron helper functions ################################
####################################################################################################

