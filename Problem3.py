from Helpers import MultiLayerPerceptron

def Problem3(x_train, y_train, x_test, y_test, greyscale_range):
	# Declare training and test sets for digits 0-9
	#all_train = [[] for _ in range(10)]
	#all_test = [[] for _ in range(10)]

	# Step 1: Divide each 28x28 image into a 7x7 grid of 4x4 squares
	# Get the average grayscale value of each square. This average will
	# be used as a feature
	# x_train and x_test are uint8 arrays of grayscale image data with
	# shape (num_samples, 28, 28).
	# y_train and y_test are uint8 arrays of digit labels 
	# (integers in range 0-9) with shape (num_samples,).
	
	set_len = 10
	#set_len = len(x_train)
	img_4x4_train_set = [[0 for _ in range(49)] for _ in range(set_len)]
	#for i in range(len(x_train)):
	for i in range(set_len):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_train[i][x][y]
				img_4x4_train_set[i][j*7//4+k//4] = total

	"""
	img_4x4_test_set = [[0 for _ in range(49)] for _ in range(len(x_test))]
	for i in range(len(x_test)):
		for j in range(0,28,4):
			for k in range(0,28,4):
				total = 0
				for x in range(j,j+4):
					for y in range(k,k+4):
						total += x_test[i][x][y]
				img_4x4_test_set[i][j*7//4+k//4] = total
	"""
	

	# Step 2: Select the number of neurons in the hidden layer
	# n = 3, 6, 10, 15, 21
	# There will be ten output neurons.
	# Perform 1000 iterations for each value of n
	#hidden_neurons = [3,6,10,15,21]
	hidden_neurons = 10
	output_neurons = 10
	epochs = 1000
	#input_set = [[0,0],[0,1],[1,0],[1,1]]
	#target_wgts = [0,0,0,1]
	perceptrons = []
	#print(img_4x4_train_set[1])
	perceptrons.append(MultiLayerPerceptron(img_4x4_train_set,  hidden_neurons, output_neurons, 
		epochs, y_train, img_4x4_train_set, y_train, set_len))
	#perceptrons.append(MultiLayerPerceptron(input_set,  hidden_neurons, output_neurons, 
		#epochs, target_wgts, input_set, target_wgts))
	#print(perceptrons[0].GetTestOutputError())
	#print(perceptrons[0].GetTestOutputActivations())
	
	"""perceptrons.append(MultiLayerPerceptron(img_4x4_train_set,  6, 10, 1000, y_train,
	img_4x4_test_set, y_test))
	perceptrons.append(MultiLayerPerceptron(img_4x4_train_set, 10, 10, 1000, y_train,
	img_4x4_test_set, y_test))
	perceptrons.append(MultiLayerPerceptron(img_4x4_train_set, 15, 10, 1000, y_train,
	img_4x4_test_set, y_test))
	perceptrons.append(MultiLayerPerceptron(img_4x4_train_set, 21, 10, 1000, y_train,
	img_4x4_test_set, y_test))

	p

	# Step 3: After each iteration, test performance will be used to determine
	# its error rate.

	min_total_error = 99999999
	min_perceptron = -1
	for i in range(len(perceptrons)):
		total_error = 0
		for j in range(10):
			total_error += perceptrons[i].GetTestOutputError()[j]
		if(total_error < min_total_error):
			min_total_error = total_error
			min_perceptron = i
	print(str(perceptrons[i].GetNumHiddenNodes()) + " hidden nodes have the min error: " 
	+ str(round(min_total_error * 100, 1)) + "%")

	# Step 4: Select the value of n with the minimum error rate.
	n = perceptrons[i].GetNumHiddenNodes()

	# Step 5: Use the best performing neural network to identify the digit in a
	# given test image
	"""