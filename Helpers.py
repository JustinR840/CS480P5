class Image:
	def __init__(self, data, pixelWidth, pixelHeight, greyscaleRange):
		self._data = data
		self._pixelWidth = pixelWidth
		self._pixelHeight = pixelHeight
		self._greyscaleRange = greyscaleRange

		self._density = -1
		self._degreeOfSymmetry = -1
		self._maxVerticalIntersections = -1
		self._avgVerticalIntersections = -1
		self._maxHorizontalIntersections = -1
		self._avgHorizontalIntersections = -1

	# 1: Density: Density is defined as the average gray scale value of all the pixels in the image
	# and is thus a real number between 0 and 255.
	def Density(self):
		if(self._density != -1):
			return self._density

		total = 0

		for i in range(len(self._data)):
			total += self._data[i]

		self._density = total / (self._pixelHeight * self._pixelWidth)

		return self._density

	# 2: Measure of symmetry is defined as the average gray scale of the image obtained by the
	# bitwise XOR (⊕) of each pixel with its corresponding vertically reflected image.
	# (Thus if I is the image, let I' be the image whose j-th column is the (28 − j)-th column of I. Then,
	# the measure of symmetry is the density of I ⊕ I'.)
	def DegreeOfSymmetry(self):
		if(self._degreeOfSymmetry != -1):
			return self._degreeOfSymmetry

		total = 0

		for i in range(self._pixelHeight):
			for j in range(self._pixelWidth):
				normal_pos = (i * self._pixelWidth) + j
				inverted_pos = (i * self._pixelWidth) + (self._pixelWidth - j - 1)

				normal_val = self._data[normal_pos]
				inverted_val = self._data[inverted_pos]

				total += normal_val ^ inverted_val

		density = total / (self._pixelWidth * self._pixelHeight)

		self._degreeOfSymmetry = density

		return self._degreeOfSymmetry

	# 3: The number of vertical intersections is defined as follows:
	# First turn the image into black and white image by assigning a color 0 (1) for gray scale values above (below) 128.
	# Consider the j-th column as a string of 0’s and 1’s and count the number of changes from 0 to 1 or 1 to 0.
	# For example, the number of changes for string 11011001 is 4.
	# Average this over all columns to get the average number of vertical intersections, and maximum over all columns will
	# give the maximum number of vertical intersections.
	# Vertical measures are defined in a similar way based on rows.
	def SetAvgAndMaxVerticalIntersections(self):
		if(self._avgVerticalIntersections != -1 and self._maxVerticalIntersections != -1):
			return

		booleans = [0 for _ in range(self._pixelWidth * self._pixelHeight)]

		for i in range(self._pixelHeight):
			for j in range(self._pixelWidth):
				pos = (i * self._pixelHeight) + j

				val = 0 if self._data[pos] >= self._greyscaleRange // 2 else 1
				booleans[pos] = val

		column_swaps = [0 for _ in range(self._pixelWidth)]
		for i in range(self._pixelWidth):
			last_val = booleans[i]

			for j in range(1, self._pixelHeight):
				pos = (j * self._pixelHeight) + i

				if(booleans[pos] != last_val):
					last_val = booleans[pos]
					column_swaps[i] += 1

		total = 0

		for i in range(len(column_swaps)):
			total += column_swaps[i]

		avg = total / len(column_swaps)

		self._maxVerticalIntersections = max(column_swaps)
		self._avgVerticalIntersections = avg


	def AvgVerticalIntersections(self):
		if(self._avgVerticalIntersections == -1):
			self.SetAvgAndMaxVerticalIntersections()
		return self._avgVerticalIntersections


	def MaxVerticalIntersections(self):
		if(self._maxVerticalIntersections == -1):
			self.SetAvgAndMaxVerticalIntersections()
		return self._maxVerticalIntersections


	def SetAvgAndMaxHorizontalIntersections(self):
		if(self._avgHorizontalIntersections != -1 and self._maxHorizontalIntersections != -1):
			return

		booleans = [0 for _ in range(self._pixelWidth * self._pixelHeight)]

		for i in range(self._pixelHeight):
			for j in range(self._pixelWidth):
				pos = (i * self._pixelHeight) + j

				val = 0 if self._data[pos] >= self._greyscaleRange // 2 else 1
				booleans[pos] = val

		row_swaps = [0 for _ in range(self._pixelHeight)]
		for i in range(self._pixelHeight):
			last_val = booleans[i * self._pixelHeight]

			for j in range(1, self._pixelWidth):
				pos = (i * self._pixelHeight) + j

				if(booleans[pos] != last_val):
					last_val = booleans[pos]
					row_swaps[i] += 1

		total = 0

		for i in range(len(row_swaps)):
			total += row_swaps[i]

		avg = total / len(row_swaps)

		self._maxHorizontalIntersections = max(row_swaps)
		self._avgHorizontalIntersections = avg


	def AvgHorizontalIntersections(self):
		if(self._avgHorizontalIntersections == -1):
			self.SetAvgAndMaxHorizontalIntersections()
		return self._avgHorizontalIntersections


	def MaxHorizontalIntersections(self):
		if(self._maxHorizontalIntersections == -1):
			self.SetAvgAndMaxHorizontalIntersections()
		return self._maxHorizontalIntersections
