def LoadData():

	import json
	import numpy as np

	x_train, y_train, x_test, y_test = [],[],[],[]

	# Build the training files
	for i in range(10):

		# Get the data from the json file
		input_file = open("train" + str(i) + ".json")
		data = json.load(input_file)
		
		# When we are rebuilding the files, the x and y sets will be in order.
		# Thus, randomizing the input to the machine learning algorithms is important
		# Build the x file and y file
		for j in range(len(data)):
			x_train.append(data[j])
			y_train.append(j)

	# Build the test files
	for i in range(10):

		# Get the data from the json file
		input_file = open("test" + str(i) + ".json")
		data = json.load(input_file)
		
		# When we are rebuilding the files, the x and y sets will be in order.
		# Thus, randomizing the input to the machine learning algorithms is important
		# Build the x file and y file
		for j in range(len(data)):
			x_test.append(data[j])
			y_test.append(j)

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	return (x_train, y_train), (x_test, y_test)
#LoadData()