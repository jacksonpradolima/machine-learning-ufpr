#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import sys

import os
import pickle

from collections import defaultdict
from sklearn.metrics import confusion_matrix

def plotBoxplotByClass(positiveByClass, negativebyClass, labels):
	plt.figure()

	bpl = plt.boxplot(positiveByClass, positions=np.array(range(len(positiveByClass)))*2.0-0.4, sym='', widths=0.6)
	bpr = plt.boxplot(negativebyClass, positions=np.array(range(len(negativebyClass)))*2.0+0.4, sym='', widths=0.6)
	set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
	set_box_color(bpr, '#2C7BB6')

	# draw temporary red and blue lines and use them to create a legend
	plt.plot([], c='#D7191C', label='Positive')
	plt.plot([], c='#2C7BB6', label='Negative')
	plt.legend()

	plt.xticks(range(0, len(labels) * 2, 2), labels)
	plt.xlim(-2, len(labels)*2)
	#plt.ylim(0, 8)
	plt.tight_layout()
	plt.show()

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plotHist(positive, negative):
	plt.figure()
	plt.title("Probabilities - Naive Bayes")
	plt.hist((positive,negative))
	plt.legend(("Positive", "Negative"))
	plt.show()

def oldPlot(classifierResults):
	values = classifierResults[0]

	positive = []
	negative = []
	for target, pr in zip(values.y_test, values.probas):
		index = pr.argmax()
		value = pr[index]
		positive.append(value) if target == index else negative.append(value)

	plotHist(positive, negative)

def main():	
	classifierResults = []
	filename = "classifier_results/Naive Bayes_results.p"
	
	classifierResults = pickle.load(open(filename, "rb" ))
	#oldPlot(classifierResults)

	test = [[]]*len(classifierResults)
	names = []

	for i, clfr in enumerate(classifierResults):
		cm = confusion_matrix(clfr.y_test, clfr.predictions)
		np.set_printoptions(precision=2)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		test[i] = np.matrix(cm).diagonal()
		names.append("{}%".format(round(clfr.weight_train*100,0)))
		#print(test[i])
		print(names[i])
		print("\n")


	plt.figure()
	plt.title("Distribuição da Matrix de Confusão - Naive Bayes")
	plt.boxplot(test, labels=names)
	plt.ylabel('Escore')
	plt.xlabel('%Treinamento')
	plt.tick_params(labelsize="medium")

	#te = []
	#for i, t in enumerate(test):
		#te.append(i//10)
	
	#plt.setp(te, rotation=45, fontsize=8)	
	#plt.xticks([range(0, len(te) * 2)], te)
	#plt.xlim(-2, len(te)*2)
	plt.show()



if __name__ == "__main__":
	main()


