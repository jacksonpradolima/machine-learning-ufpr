import matplotlib.pyplot as plt
import numpy as np

from scikitplot import plotters as skplt
from sklearn.metrics import confusion_matrix

class PlotUtils():
	def __init__(self):
		self.a = "1"

	def plotMeanAccuracyWithTrainVariance(self, models_results):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

		self._plotMeanAccuracyWithTrainVarianceLines(models_results, ax1)
		self._plotMeanAccuracyWithTrainVarianceBoxplot(models_results, ax2)
		
		plt.tight_layout()
		fig.subplots_adjust(top=0.8)
		fig.suptitle("Comparação entre Classificadores")
		plt.savefig("plots/mean_accurancy_among_models_train_variance.pdf")

	def _plotMeanAccuracyWithTrainVarianceLines(self, models_results, ax):
		ax.set_title("Comparação utilizando gráfico de linha")

		for key, values in models_results.items():
			score = np.zeros(len(values), dtype=np.float64)
			weight = np.zeros(len(values), dtype=np.float64)
			
			for i, classifier_result in enumerate(values):
				score[i] = classifier_result.score
				weight[i] = classifier_result.weight_train

			ax.plot(weight, score, label=key)

		ax.legend(loc='upper right')
		ax.set_ylabel('Escore')
		ax.set_xlabel('%Treinamento')
		ax.grid(True)
		ax.tick_params(labelsize="medium")

		return ax

	def _plotMeanAccuracyWithTrainVarianceBoxplot(self, models_results, ax):
		ax.set_title("Comparação utilizando gráfico de caixa")

		scores = []
		names = []
		for key, values in models_results.items():
			score = np.zeros(len(values), dtype=np.float64)
			
			for i, classifier_result in enumerate(values):
				score[i] = classifier_result.score

			scores.append(score)
			names.append(key)

		ax.boxplot(scores)
		ax.set_xticklabels(names)

		return ax

	def plotConfusionMatrixBoxplot(self, models_results):
		for key, values in models_results.items():
			
			matrix_dist = [[]]*len(values)
			names = []

			for i, clfr in enumerate(values):
				cm = confusion_matrix(clfr.y_test, clfr.predictions)
				np.set_printoptions(precision=2)
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				matrix_dist[i] = np.matrix(cm).diagonal()
				names.append("{}%".format(round(clfr.weight_train*100,0)))

			plt.figure()
			plt.title("Distribuição da Matrix de Confusão - " + key)
			plt.boxplot(matrix_dist, labels=names)
			plt.ylabel('Escore')
			plt.xlabel('%Treinamento')
			plt.tick_params(labelsize="medium")
			plt.tight_layout()
			plt.savefig("plots/confusion_matrix_distribution_" + key + ".pdf")

	def plotConfusionMatrix(self, models_results):
		for key, values in models_results.items():
			fig, axes = plt.subplots(3, 3, figsize=(15,15))

			indexes = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

			for i, clfr in enumerate(values):
				skplt.plot_confusion_matrix(y_true=clfr.y_test, y_pred=clfr.predictions, normalize=True, ax=axes[indexes[i]], title="Matrix de Confusão Normalizada")

				plt.sca(axes[indexes[i]])
				axes[indexes[i]].set_xlabel("Treinamento/Teste ({}/{})".format(round(clfr.weight_train*100,0), round(clfr.weight_test*100,0)))  # set x label
				axes[indexes[i]].get_xaxis().set_ticks([])       # hidden x axis text
				axes[indexes[i]].get_yaxis().set_ticks([])
			
			plt.tight_layout()			
			# fig.subplots_adjust(hspace=0.3)
			fig.subplots_adjust(top=0.95)
			fig.suptitle(key, fontsize=16)
			plt.savefig("plots/confusion_matrix_{}.pdf".format(key))

	def plotProbabilityDistribution(self, models_results_50_50):
		for key, values in models_results_50_50.items():
			plt.figure()
			plt.title("Distribuição de Probabilidade - " + key)

			labels = np.unique(values.y_test)
			positive = []
			negative = []
			positiveByClass = [[]]*len(labels)
			negativebyClass = [[]]*len(labels)	

			for target, pr in zip(values.y_test, values.probas):
				index = pr.argmax()
				value = pr[index]
				
				if target == index:			
					positive.append(value)
					positiveByClass[index].append(value)
				else:			
					negative.append(value)
					negativebyClass[index].append(value)

			plt.hist((positive,negative))			
			plt.legend(("Positivo", "Negativo"))
			plt.tight_layout()
			plt.savefig("plots/probability_distribution_" + key + ".pdf")

	def plotRecallPrecision(self, models_results):
		for key, values in models_results.items():
			fig, axes = plt.subplots(3, 3, figsize=(15,15))

			indexes = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

			for i, clfr in enumerate(values):
				skplt.plot_precision_recall_curve(y_true=clfr.y_test, y_probas=clfr.probas, ax=axes[indexes[i]])

				plt.sca(axes[indexes[i]])
				axes[indexes[i]].set_xlabel("Training/Test ({}/{})".format(round(clfr.weight_train*100,0), round(clfr.weight_test*100,0)))  # set x label
				axes[indexes[i]].get_xaxis().set_ticks([])       # hidden x axis text
				axes[indexes[i]].get_yaxis().set_ticks([])

			plt.tight_layout()
			fig.subplots_adjust(top=0.95)
			fig.suptitle(key, fontsize=16)
			plt.savefig("plots/precision_recall_curve_{}.pdf".format(key))

	def plotRocCurve(self, classifier_name, classifierResults):
		fig, axes = plt.subplots(3, 3, figsize=(15,15))
		fig.canvas.set_window_title(classifier_name)

		indexes = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

		for i, classifierResult in enumerate(classifierResults):

			skplt.plot_roc_curve(y_true=classifierResult.y_test, y_probas=classifierResult.probas, ax=axes[indexes[i]])

			# set the current axes instance 
			plt.sca(axes[indexes[i]])
			axes[indexes[i]].set_xlabel("Training/Test ({}/{})".format(round(classifierResult.weight_train*100,0), round(classifierResult.weight_test*100,0)))  # set x label
			axes[indexes[i]].get_xaxis().set_ticks([])       # hidden x axis text
			axes[indexes[i]].get_yaxis().set_ticks([])

		fig.subplots_adjust(hspace=0.3)
		plt.tight_layout()
		plt.savefig("plots/roc_curve_{}.pdf".format(classifier_name))

	def plotRocCurve50_50(self, models_results):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))

		i = 0
		for key, values in models_results.items():
			self._plotRocCurve50_50(key, values, ax1 if i == 0 else ax2)
			i = 1
		
		plt.tight_layout()
		fig.subplots_adjust(top=0.95)
		fig.suptitle("Classifier Comparison (50/50)")
		plt.savefig("plots/roccurve_train_50.pdf")

	def _plotRocCurve50_50(self, key, values, ax):
		for i, clfr in enumerate(values):
			if(clfr.weight_train == 0.5):
				skplt.plot_roc_curve(y_true=clfr.y_test, y_probas=clfr.probas, ax=ax, title=key)

		ax.legend(loc='lower right')
		ax.set_ylabel('Score')
		ax.set_xlabel('%Train')
		ax.grid(True)
		ax.tick_params(labelsize="medium")

		return ax
	def plotMetrics(self, models_results):
		plt.figure()
		plt.title("Metrics among models")
		for key, values in models_results.items():
			accuracy = np.zeros(len(values), dtype=np.float64)
			precision = np.zeros(len(values), dtype=np.float64)
			recall = np.zeros(len(values), dtype=np.float64)
			f1 = np.zeros(len(values), dtype=np.float64)
			weight_train = np.zeros(len(values), dtype=np.float64)
			for i, classifier_result in enumerate(values):
				accuracy[i] = classifier_result.accuracy
				precision[i] = classifier_result.precision
				recall[i] = classifier_result.recall
				f1[i] = classifier_result.f1
				weight_train[i] = classifier_result.weight_train

			plt.plot(weight_train, accuracy, label=key+" - Accuracy")
			plt.plot(weight_train, precision, label=key+" - Precision")
			plt.plot(weight_train, recall, label=key+" - Recall")
			plt.plot(weight_train, f1, label=key+" - F1")

		plt.legend(loc='upper right')
		plt.ylabel('Score')
		plt.xlabel('%Train')
		plt.grid(True)
		plt.tick_params(labelsize="medium")
		plt.savefig("plots/metrics_train_among_models.pdf")



	# def plotMeanAccuracy(weights_test,weights_train,scores, title):
	# 	plt.figure()	
	# 	plt.title("Mean Accuracy - {}".format(title))
	# 	plt.plot(weights_test, scores, 'red', label='Test - Mean Accuracy')
	# 	plt.plot(weights_train, scores, 'darkblue', label='Training - Mean Accuracy')
	# 	plt.legend(loc='upper right')
	# 	plt.ylabel('Score')
	# 	plt.xlabel('%')
	# 	plt.grid(True)
	# 	plt.tick_params(labelsize="medium")
	# 	plt.savefig("plots/mean_accurancy_{}.pdf".format(title))	