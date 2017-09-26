#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================
Perceptron Algorithm
=====================

Perceptron classifier made with Python3. 

For each epoch is generated a plot, showed, and save in plot folder. 
"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import random, os, shutil, sys

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "GPL"
__version__ = "1.0"

def generate_data(no_points):
	"""
	Generates a 2D linearly separable dataset with 'no_points' samples

	Parameters
	------------
	no_points: int
		Number of points
	"""
	X = np.zeros(shape=(no_points, 2))
	Y = np.zeros(shape=no_points)

	for i in range(no_points):
		X[i][0] = random.randint(-9,9)+0.5
		X[i][1] = random.randint(-9,9)+0.5
		Y[i] = 1 if X[i][0]+X[i][1] >= 2 else 0

	return X, Y

def predict(row_X, weights, bias):
	"""
	Activation function
	"""
	return 1 if weights.dot(row_X) + bias >= 0 else 0

def accuracy(X, Y, weights, bias):
	"""
	Evaluates the accuracy of the weights and bias
	"""
	sum_correct = 0.0

	for row_X, row_Y in zip(X,Y):
		if row_Y == predict(row_X, weights, bias): sum_correct +=1.0

	return sum_correct/float(len(Y))

def plot(X, Y, weights, bias, epoch):
	"""
	Plots a epoch
	"""
	fig,ax = plt.subplots()
	ax.set_title("Epoch %d" % epoch)
	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
		
	xx = [-10,10]
	ax.set_xlim(xx)
	ax.set_ylim(xx)

	plt.plot([0,0], xx, linewidth=0.5, c='gray')
	plt.plot(xx, [0,0], linewidth=0.5, c='gray')

	# Class division
	c1_data=[[],[]]
	c0_data=[[],[]]
	for row_X, row_Y in zip(X,Y):
		cur_i1 = row_X[0]
		cur_i2 = row_X[1]

		if row_Y == 1:
			c1_data[0].append(cur_i1)
			c1_data[1].append(cur_i2)
		else:
			c0_data[0].append(cur_i1)
			c0_data[1].append(cur_i2)

	# plot the classes
	c0s = plt.scatter(c0_data[0],c0_data[1],s=10.0,c='r',label='Class 0')
	c1s = plt.scatter(c1_data[0],c1_data[1],s=10.0,c='b',label='Class 1')

	# Plot the line
	yy = (-np.dot(weights[0], xx) - bias) / weights[1]
	plt.plot(xx, yy, linewidth=0.6, c='black')

	plt.tight_layout()
	plt.legend(bbox_to_anchor=(0.5, -0.1), fontsize=10,loc='upper center', ncol=2)
	plt.savefig('plots/epoch_%s' % (str(epoch)), dpi=200, bbox_inches='tight')
	plt.show()
	return

def train(X, Y, learning_rate, nb_epoch):
	"""
	Train the weights

	Parameters
	------------
	learning_rate: float
		Used to limit the amount each weight is corrected each time it is updated
	nb_epoch: float
		The number of times to run through the training data while updating the weight.
	"""
	# deslocamento em relação a origem
	bias = 0

	# Gero os pesos aleatoriamente
	weights = np.random.rand(2)

	best_acc = -1

	# Número de vezes (máximo de iterações/épocas) que irei ajustar os pesos
	for epoch in range(1, nb_epoch+1):
		cur_acc = accuracy(X, Y, weights, bias)
		print('> epoch=%d, bias=%.3f, weights=%s, accuracy=%f' % (epoch, bias, weights, cur_acc))

		# Guardo a melhor época
		if cur_acc > best_acc:
			best_weights = weights
			best_epoch = epoch
			best_acc = cur_acc

		# Cria um gráfico para cada época
		plot(X, Y, weights, bias, epoch)

		# Se eu encontrei os pesos que refletem a melhor acurácia possível (100%) eu paro as iterações
		if cur_acc == 1.0:
			print("\nAccurracy equals 100%! Stopping Perceptron...")
			break

		for row_X, row_Y in zip(X,Y):
			#calculo a predição e o erro
			error = row_Y - predict(row_X, weights, bias)

			if error != 0:
				update = learning_rate * error

				# Atualizar os pesos
				weights += update * row_X

				# Atualizar o bias
				bias += update
	
	print('\n--- training result --- \n> epoch=%d, bias=%.3f, weights=%s, accuracy=%f \n-----------------------\n' % (best_epoch, bias, best_weights, best_acc))

	return best_weights, bias, best_epoch, best_acc

def main(nb_epoch):
	basedir = os.getcwd() + "/plots"

	if os.path.isdir("plots"):
		shutil.rmtree(basedir)

	os.makedirs("plots")

	print("Preparing data...\nGenerating a new linearly separable dataset...")

	X, Y = generate_data(100)
	X_test, Y_test = generate_data(50)

	print("Training...\n")
	weights, bias, epoch, acc = train(X, Y, 0.1, int(nb_epoch))

	# Realizo um segundo teste 
	print("\nEvalutating weights and bias to a new linearly separable dataset... \nAccuracy: %f" % accuracy(X_test, Y_test, weights, bias))

	print("\nFinished!")
if __name__ == '__main__':
	if len(sys.argv) != 2:
		sys.exit("Use: main.py <nb_epoch>")

	main(sys.argv[1])