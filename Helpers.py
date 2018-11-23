def AllWeightsValid(perceptron, inputs, targets):
	for i in range(len(inputs)):
		if(perceptron.ActivationValue(inputs[i]) != targets[i]):
			return False
	return True


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
		# Convert the entire array to boolean values
		booleans = [[0 for _ in range(self.__pixelWidth)] for _ in range(self.__pixelHeight)]
		for row in range(self.__pixelHeight):
			for col in range(self.__pixelWidth):
				val = 0 if self.__image[row][col] >= self.__greyscaleRange // 2 else 1
				booleans[row][col] = val

		# Go over the boolean values and count the number of times the values swap
		# from 0->1 or 1->0 in the COLUMNS (vertical)
		# This loop goes over the image using COLUMN MAJOR ORDER
		column_swaps = [0 for _ in range(self.__pixelWidth)]
		for col in range(self.__pixelWidth):
			last_val = booleans[0][col]

			for row in range(1, self.__pixelHeight):
				if(booleans[row][col] != last_val):
					last_val = booleans[row][col]
					column_swaps[col] += 1

		# Calculate the average vertical intersections and max vertical intersections
		total = 0
		for i in range(len(column_swaps)):
			total += column_swaps[i]
		avg = total / len(column_swaps)

		self.__maxVerticalIntersections = max(column_swaps)
		self.__avgVerticalIntersections = avg

	def __SetAvgAndMaxHorizontalIntersections(self):
		# Convert the entire array to boolean values
		booleans = [[0 for _ in range(self.__pixelWidth)] for _ in range(self.__pixelHeight)]
		for row in range(self.__pixelHeight):
			for col in range(self.__pixelWidth):
				val = 0 if self.__image[row][col] >= self.__greyscaleRange // 2 else 1
				booleans[row][col] = val

		# Go over the boolean values and count the number of times the values swap
		# from 0->1 or 1->0 in the ROWS (horizontal)
		# This loop goes over the image using ROW MAJOR ORDER
		row_swaps = [0 for _ in range(self.__pixelHeight)]
		for row in range(self.__pixelHeight):
			last_val = booleans[row][0]

			for col in range(1, self.__pixelWidth):
				if(booleans[row][col] != last_val):
					last_val = booleans[row][col]
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
