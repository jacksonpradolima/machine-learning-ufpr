from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import numpy as np
import csv

if __name__ == '__main__':
	X_test, y_test = load_svmlight_file('test.dat')

	with open('test.dat.predict') as f:
		predictions = f.readlines()

	predictions = [float(x.strip()) for x in predictions]

	skplt.plot_confusion_matrix(y_true=y_test, y_pred=predictions, normalize=True, title="Matrix de Confusão Normalizada")

	plt.savefig('confusion-matrix_1.pdf', bbox_inches='tight')

	with open('prob/test.dat.prob_temp.predict') as textFile:
		predictions2 = [line.split() for line in textFile]

	plt.figure()
	plt.title("Distribuição de Probabilidade - SVM")

	positive = []
	negative = []

	for target, pr in zip(y_test, predictions2):
		predict = pr[0]
		aux = np.array([float(x) for x in pr[1:11]])
		value = float(aux[aux.argmax()])
		if int(target) == int(predict):
			positive.append(value)
		else:
			negative.append(value)

	plt.hist((positive,negative))
	plt.legend(("Acerto", "Erro"))
	plt.tight_layout()
	plt.savefig("probability_distribution_svm.pdf")

	predictions = []
	with open('prob/test.dat.prob_temp.predict') as f:
		predictions = [float(line.split()[0]) for line in f]

	skplt.plot_confusion_matrix(y_true=y_test, y_pred=predictions, normalize=True, title="Matrix de Confusão Normalizada")

	plt.savefig('confusion-matrix_2.pdf', bbox_inches='tight')