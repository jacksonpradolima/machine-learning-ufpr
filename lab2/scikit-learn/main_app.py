#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison among Decision Tree, Naive Bayers, and SVM classifiers using scikit-learn on synthetic dataset.
"""
print(__doc__)

import numpy as np
import os
import pickle
import sys

from ClassifierResult import ClassifierResult
from ClassifierUtils import ClassifierUtils
from PlotUtils import PlotUtils

from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main(description, datatr, datats):
	if not os.path.isdir("plots_"+description):
		os.makedirs("plots_"+description)

	names = ["Decision Tree", "Naive Bayes", "SVM", "SVM GridSearch"]
	classifiers = [DecisionTreeClassifier(), BernoulliNB(), SVC(kernel='linear', C=1, probability=True), SVC(probability=True)]

	classifierUtils = ClassifierUtils()

	filename = "results_{}.p".format(description)

	# I just load the data if I don't have the result for both classifiers
	models_results = {}

	if os.path.exists(filename):
		print("Loading previous results...")
		models_results = pickle.load(open(filename, "rb" ))
	else:
		print("Loading data...")
		classifierUtils.loadData(datatr, datats)

		print("Executing classifiers...")
		for name, classifier in zip(names, classifiers):
			print("Executing " + name)

			predictions, probas, score = classifierUtils.runClassifier(classifier, name)

			classifierResult = ClassifierResult(
				classifierUtils.X_train,
				classifierUtils.X_test,
				classifierUtils.y_train,
				classifierUtils.y_test,
				predictions,
				probas,
				score
			)

			if "SVM GridSearch" == name:
				classifierResult.updateBestParam(classifierUtils.getBestParam())

			models_results[name] = classifierResult

		# Save results
		pickle.dump(models_results, open(filename, "wb"))

	print("All classifiers were executed.")
	print("Plotting...")
	plotUtils = PlotUtils(description)

	sorted(models_results)
	plotUtils.plotMeanAccuracy(models_results)
	plotUtils.plotConfusionMatrix(models_results)
	plotUtils.plotConfusionMatrixBoxplot(models_results)
	plotUtils.plotProbabilityDistribution(models_results)

	print("Finished!")

if __name__ == "__main__":
	if len(sys.argv) != 4:
		sys.exit("Use: main_app.py <description> <tr> <ts>")

	main(sys.argv[1], sys.argv[2], sys.argv[3])