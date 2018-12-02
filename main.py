import tensorflow as tf
from Helpers import TestSimpleData, TransformData, DumpData
from Problem1 import Problem1
from Problem2 import Problem2
from Problem3 import Problem3
import json



def main():
	p_width = 28
	p_height = 28
	greyscale_range = 255

	# Just to verify that the internals of Image still work as expected
	#TestSimpleData(p_width, p_height, greyscale_range)

	# Get the MNIST
	mnist = tf.keras.datasets.mnist

	# Load in the MNIST data set
	# x_train and x_test are uint8 arrays of grayscale image data with
	# shape (num_samples, 28, 28).
	# y_train and y_test are uint8 arrays of digit labels 
	# (integers in range 0-9) with shape (num_samples,).
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#print(len(y_test))
	Problem1(x_train, y_train, p_width, p_height, greyscale_range)
	#Problem2(x_train, y_train, p_width, p_height, greyscale_range)
	#Problem3(x_train, y_train, x_test, y_test, greyscale_range, 10)
	#run_perceptron(x_train, y_train, x_test, y_test, 5)


if __name__ == '__main__':
	main()