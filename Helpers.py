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