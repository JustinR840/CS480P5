import tensorflow as tf
from Helpers import TestSimpleData
from Problem1 import Problem1
from Problem2 import Problem2



def main():
	p_width = 28
	p_height = 28
	greyscale_range = 255

	# Just to verify that the internals of Image still work as expected
	TestSimpleData(p_width, p_height, greyscale_range)





	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()


	#Problem1(x_train, y_train, p_width, p_height, greyscale_range)
	Problem2(x_train, y_train, p_width, p_height, greyscale_range)




















main()