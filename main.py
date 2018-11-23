from Helpers import Image


def main():
	p_width = 28
	p_height = 28
	greyscale_range = 255

	TestSimpleData(p_width, p_height, greyscale_range)



















def TestSimpleData(p_width, p_height, greyscale_range):
	# [0  , 0  ]
	# [255, 255]
	# The following data should have:
	#	degreeOfSymmetry = 0
	#	maxVerticalIntersections = 1
	#	avgVerticalIntersections = 1
	#	maxHorizontalIntersections = 0
	#	avgHorizontalIntersections = 0
	data1 = [0 if x < ((p_width * p_height) // 2) else greyscale_range for x in range(p_width * p_height)]

	# [0, 255]
	# [0, 255]
	# The following data should have:
	#	degreeOfSymmetry = 255
	#	maxVerticalIntersections = 0
	#	avgVerticalIntersections = 0
	#	maxHorizontalIntersections = 1
	#	avgHorizontalIntersections = 1
	data2 = [0 if (x % p_height) < (p_width // 2) else greyscale_range for x in range(p_width * p_height)]


	img1 = Image(data1, p_width, p_height, greyscale_range)
	img2 = Image(data2, p_width, p_height, greyscale_range)


	assert(img1.DegreeOfSymmetry() == 0)
	assert(img1.MaxVerticalIntersections() == 1)
	assert(img1.AvgVerticalIntersections() == 1)
	assert(img1.MaxHorizontalIntersections() == 0)
	assert(img1.AvgHorizontalIntersections() == 0)

	assert(img2.DegreeOfSymmetry() == greyscale_range)
	assert(img2.MaxVerticalIntersections() == 0)
	assert(img2.AvgVerticalIntersections() == 0)
	assert(img2.MaxHorizontalIntersections() == 1)
	assert(img2.AvgHorizontalIntersections() == 1)










main()