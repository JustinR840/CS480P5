from Helpers import LoadInputs
from Problem1 import Problem1
from Problem2 import Problem2
from Problem3 import Problem3
from CreateMNISTDataSets import CreateFiles
from LoadMNISTDataSets import LoadData



def main():
	greyscale_range = 255

	# Run this to build the files
	#CreateFiles()

	(x_train, y_train), (x_test, y_test) = LoadData()

	train_sets = LoadInputs(x_train, y_train)
	test_sets = LoadInputs(x_test, y_test)

	print("Problem 1")
	Problem1(train_sets[7], train_sets[9], test_sets[7], test_sets[9])

	print("Problem 2")
	Problem2(test_sets, test_sets)

	print("Problem 3")
	prob3s = []
	prob3s.append(Problem3(x_train, y_train, x_test, y_test, greyscale_range, 3))
	prob3s.append(Problem3(x_train, y_train, x_test, y_test, greyscale_range, 6))
	prob3s.append(Problem3(x_train, y_train, x_test, y_test, greyscale_range, 10))
	prob3s.append(Problem3(x_train, y_train, x_test, y_test, greyscale_range, 15))
	prob3s.append(Problem3(x_train, y_train, x_test, y_test, greyscale_range, 21)) 

	# Report findings for optimal network
	best_prob3 = prob3s[0]
	for p3 in prob3s:
		if(p3["error_rate"] < best_prob3["error_rate"]):
			best_prob3 = p3

	print(str(best_prob3["num_hidden_nodes"]) + " hidden layer neurons gave the minimum error rate")
	print("Epoch with the minimum error is " + str(best_prob3["epoch_min_error"]))
	print("Error rate is " + str(best_prob3["error_rate"]))
	print("Hidden layer neuron weights:")
	print(best_prob3["hidden_wgts"])
	print("Output layer neuron weights:")
	print(best_prob3["output_wgts"])
	print("Confusion matrix:")
	print("X axis is the actual value")
	print("Y axis is the guessed value")
	print(best_prob3["confusion_matrix"])

if __name__ == '__main__':
	main()