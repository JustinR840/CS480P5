from Helpers import Image, MulticlassPerceptron, AllWeightsValid
from random import randint, shuffle


def Problem2(x_train, y_train, p_width, p_height, greyscale_range):
	all_train = [[] for _ in range(10)]
	all_test = [[] for _ in range(10)]

	for i in range(len(y_train)):
		if (i % 100 == 0):
			print("Transformed", i, "entries into Images")

		all_train[int(y_train[i])].append(Image(x_train[i], y_train[i], p_width, p_height, greyscale_range))

	for i in range(len(all_train)):
		all_train[i] = all_train[i][:round(len(all_train[i]) * 0.8)]
		all_test[i] = all_train[i][:round(len(all_train[i]) * -0.2)]

	training_data = []
	for i in range(len(all_train)):
		training_data = training_data + all_train[i]
	shuffle(training_data)

	train_inputs = []
	train_targets = []
	for i in range(len(training_data)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TRAIN set")

		m_input = []
		m_input.append(training_data[i].GetDensity())
		m_input.append(training_data[i].GetDegreeOfSymmetry())
		m_input.append(training_data[i].GetAvgVerticalIntersections())
		m_input.append(training_data[i].GetMaxVerticalIntersections())
		m_input.append(training_data[i].GetAvgHorizontalIntersections())
		m_input.append(training_data[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		train_inputs.append(m_input)

		train_targets.append(training_data[i].GetNumber())

	#weights = [[randint(-10, 11) * 0.01 for _ in range(len(train_inputs[0]))] for _ in range(10)]
	weights = [[14.540459183662914, 2.732959183685719, 157.93714285704223, -191.36999999936046, 1378.0550000039402, -127.61000000031657, 2266.0499999990348], [-18.618596938776985, 14.494948979596817, -590.9057142882646, -118.55000000087142, -493.2350000000506, -44.13999999996542, -2482.839999998838], [4.86839285715719, 1.7550510203812477, 252.20000000032095, 4.599999999959085, -472.5699999965606, 93.28999999880652, -107.68999999999815], [1.1632397959082468, -2.5724999999607507, 316.4514285687076, 175.59999999878875, -597.6149999930509, -36.010000000599064, 281.6699999999956], [1.6505357142912467, -6.922908163271177, -244.4950000000239, -47.80999999995092, 398.36071428369183, 17.24999999988911, -144.41999999999607], [-9.844413265312578, 5.584948979598018, 402.78071428542677, 66.24000000045395, -792.3735714302169, 5.240000000010438, -611.4600000000705], [2.1704846938800735, -3.440663265363023, -135.02928571420009, 23.38000000007501, 132.20285714326056, 116.64999999879255, 419.96000000002704], [-3.991301020428641, -8.682908163275863, -17.14428571435796, -92.45999999793425, -464.53428571967055, 40.22000000014764, -1605.969999999635], [9.094158163297985, 0.8854591837130119, -46.66571428571026, 155.81000000363235, 736.3921428396767, -86.38000000024613, 2031.609999999248], [-0.88295918365184, -3.9443877551067823, -95.38928571464008, 24.59999999990675, 175.26714285733772, 21.539999999847844, -46.88000000000039]]
	eta = 0.1
	n_epochs = 1000

	p = MulticlassPerceptron(weights, eta)


	# stop = 0
	# while (stop < n_epochs):# and not AllWeightsValid(p, train_inputs, train_targets)):
	# 	if (stop % 100 == 0):
	# 		print("Gone through", stop, "epochs on the TRAIN set")
	# 	for i in range(len(train_inputs)):
	# 		activation = p.ActivationLabel(train_inputs[i])
	#
	# 		if(activation != train_targets[i]):
	# 			p.UpdateWeights(train_inputs[i], activation, train_targets[i])
	#
	# 	stop += 1

	print(p.weights)

	test_data = []
	for i in range(len(all_test)):
		test_data = test_data + all_test[i]
	shuffle(test_data)

	test_inputs = []
	test_targets = []
	for i in range(len(test_data)):
		if (i % 100 == 0):
			print("Processed ", i, "entries in the TEST set")

		m_input = []
		m_input.append(test_data[i].GetDensity())
		m_input.append(test_data[i].GetDegreeOfSymmetry())
		m_input.append(test_data[i].GetAvgVerticalIntersections())
		m_input.append(test_data[i].GetMaxVerticalIntersections())
		m_input.append(test_data[i].GetAvgHorizontalIntersections())
		m_input.append(test_data[i].GetMaxHorizontalIntersections())
		m_input.append(-1)
		test_inputs.append(m_input)

		test_targets.append(test_data[i].GetNumber())

	right = [0 for _ in range(10)]
	wrong = [0 for _ in range(10)]
	total = [0 for _ in range(10)]

	for i in range(len(test_inputs)):
		total[test_targets[i]] += 1

		if (i % 100 == 0):
			print("Gotten results for ", i, "entries in the TEST set")

		activation = p.ActivationLabel(test_inputs[i])

		if(activation == test_targets[i]):
			right[test_targets[i]] += 1
		else:
			wrong[test_targets[i]] += 1

	print("Right")
	for i in range(len(total)):
		print(str(i) + ": " + str(round((right[i] / total[i]) * 100)) + "%")

	print("Wrong")
	for i in range(len(total)):
		print(str(i) + ": " + str(round((wrong[i] / total[i]) * 100)) + "%")
