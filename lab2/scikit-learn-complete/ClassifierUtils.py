import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

class ClassifierUtils:
	def __init__(self):
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.best_params_ = None

	def loadData(self, datatr, datats):
		self.X_train, self.y_train = load_svmlight_file(datatr)
		self.X_test, self.y_test = load_svmlight_file(datats)

	def GridSearch(self, model):
		# define range dos parametros
		C_range = 2. ** np.arange(-5,15,2)
		gamma_range = 2. ** np.arange(3,-15,-2)
		k = [ 'rbf']

		#k = ['linear', 'rbf']
		param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)
		
		# faz a busca
		grid = GridSearchCV(model, param_grid, n_jobs=-1, verbose=True)
		grid.fit(self.X_train, self.y_train)

		# recupera o melhor modelo
		best = grid.best_estimator_

		# guarda os parametros desse modelo
		self.best_params_ = grid.best_params_

		return best

	def getBestParam(self):
		return self.best_params_

	def runClassifier(self, model, name):
		model = self.GridSearch(model) if "SVM GridSearch" == name else model

		# Train model.
		model = model.fit(self.X_train, self.y_train)

		# Make predictions.
		predictions = model.predict(self.X_test) 

		# Returns the mean accuracy on the given test data and labels.
		score = model.score(self.X_test, self.y_test);

		#Predict class probabilities of the input samples X_test.
		probas = model.predict_proba(self.X_test)

		return predictions, probas, score