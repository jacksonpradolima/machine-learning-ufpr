import matplotlib.pyplot as plt
import numpy as np

from scikitplot import plotters as skplt
from sklearn.metrics import confusion_matrix

class PlotUtils():
	def __init__(self, description):
		self.description = "plots_"+ description

	def plotMeanAccuracy(self, models_results):

		scores = []
		names = []

		for key, values in sorted(models_results.items()):
			scores.append(values.score)
			names.append(key)
		
		y_pos = np.arange(len(names))

		plt.figure()
		plt.bar(y_pos, scores, align='center', alpha=0.3, color='g')
		plt.xticks(y_pos, names)
		plt.title("Acurácia Média Obtida")
		plt.ylabel('Escore')
		plt.tick_params(labelsize="medium")
		plt.tight_layout()
		plt.savefig(self.description + "/mean_accurancy.pdf")

	def plotConfusionMatrixBoxplot(self, models_results):
		matrix_dist = [[]]*len(models_results)
		names = []
		i = 0
		for key, values in sorted(models_results.items()):
			cm = confusion_matrix(values.y_test, values.predictions)
			np.set_printoptions(precision=2)
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			matrix_dist[i] = np.matrix(cm).diagonal()
			names.append(key)
			i+=1

		plt.figure()
		plt.title("Distribuição das Matrizes de Confusão")
		plt.boxplot(matrix_dist, labels=names)
		plt.grid(True)
		plt.ylabel('Escore')
		plt.tick_params(labelsize="medium")
		plt.tight_layout()
		plt.savefig(self.description + "/distribution_confusion_matrix.pdf")

	def plotConfusionMatrix(self, models_results):
		for key, values in sorted(models_results.items()):
			fig, ax = plt.subplots()
			
			skplt.plot_confusion_matrix(y_true=values.y_test, y_pred=values.predictions, normalize=True, ax=ax, title="Matrix de Confusão Normalizada")

			plt.sca(ax)
			ax.set_xlabel("")  # set x label
			ax.get_xaxis().set_ticks([])  # hidden x axis text
			ax.get_yaxis().set_ticks([])
			
			plt.tight_layout()
			#fig.subplots_adjust(top=0.95)
			plt.savefig(self.description + "/confusion_matrix_{}.pdf".format(key))

	def plotProbabilityDistribution(self, models_results):
		for key, values in sorted(models_results.items()):
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
			plt.legend(("Acerto", "Erro"))
			plt.tight_layout()
			plt.savefig(self.description + "/probability_distribution_" + key + ".pdf")