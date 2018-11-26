import tensorflow as tf
import numpy as np
from Helpers import TestSimpleData
from Problem1 import Problem1
from Problem2 import Problem2
from Problem3 import Problem3



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


	#Problem1(x_train, y_train, p_width, p_height, greyscale_range)
	#Problem2(x_train, y_train, p_width, p_height, greyscale_range)
	Problem3(x_train, y_train, p_width, p_height, greyscale_range)




















main()