#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison among Decision Tree and Naive Bayers classifiers using scikit-learn on synthetic datasets.
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
from sklearn.tree import DecisionTreeClassifier

def main(data):
	if not os.path.isdir("classifier_results"):
		os.makedirs("classifier_results")

	if not os.path.isdir("plots"):
		os.makedirs("plots")

	names = ["Decision Tree", "Naive Bayes"]
	classifiers = [DecisionTreeClassifier(), BernoulliNB()]

	classifierUtils = ClassifierUtils(data)

	# I just load the data if I don't have the result for both classifiers
	if not(os.path.exists("classifier_results/Decision Tree_results.p") and os.path.exists("classifier_results/Naive Bayes_results.p")):
		print("Loading data.")
		classifierUtils.loadData()

	weights_train = np.arange(0.1, 1, 0.1)
	weights_test = weights_train[::-1]

	models_scores = {}
	models_results = {}
	models_results_50_50 = {}

	print("Executing classifiers.")
	for name, classifier in zip(names, classifiers):
		classifierResults = []
		filename = "classifier_results/{}_results.p".format(name)

		if os.path.exists(filename):
			# Avoid execute if I already executed
			classifierResults = pickle.load(open(filename, "rb" ))

			for classifierResult in classifierResults:
				if(classifierResult.weight_train == 0.5):
					models_results_50_50[name] = classifierResult
		else:
			for i, weight_test in enumerate(weights_test):
				classifierUtils.initTrainingTestingSubsets(weight_test)
				predictions, probas, score = classifierUtils.runClassifier(classifier)

				classifierResult = ClassifierResult(
					classifierUtils.X_train,
					classifierUtils.X_test,
					classifierUtils.y_train,
					classifierUtils.y_test,
					predictions,
					probas,
					score,
					weights_train[i],
					weights_test[i]
				)

				classifierResults.append(classifierResult)

				if(weights_test == 0.5):
					models_results_50_50[name] = classifierResult

			# Save results
			pickle.dump(classifierResults, open(filename, "wb"))
		# It will be used to plot
		models_results[name] = classifierResults

	print("All classifiers were executed.")
	print("Plotting.")
	plotUtils = PlotUtils()
	
	#1) utilize 10% para treinamento e 90% para teste e aumente em 10 pontos percentuais a base de treinamento (diminuindo em 10 pp a base de teste) e analise os impactos na taxa de reconhecimento.	
	# plotUtils.plotMeanAccuracyWithTrainVariance(models_results)
	
	# # 2) Analise os impactos do tamanho da base na matriz de confusão. A matriz continua com a mesma distribuição? O aumento da base diminuiu a confusão entre algumas classes ou não ? Os dois classificadores tem o mesmo comportamento?
	# plotUtils.plotConfusionMatrix(models_results)
	# plotUtils.plotRecallPrecision(models_results)

	# 3) Utilizando o método que estima probabilidade, mostre a distribuição das probabilidades para os acertos e erros de cada classificador. Compare as distribuições dos dois classificadores e reporte as suas conclusões. Para esse experimento utilize (50-50 para treinamento e teste).
	plotUtils.plotProbabilityDistribution(models_results_50_50)

	print("Finished!")

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Use: svm.py <data>")

	main(sys.argv[1])