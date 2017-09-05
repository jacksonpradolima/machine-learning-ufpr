#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import numpy
import sys

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

def GridSearch(X_train, y_train):

	# define range dos parametros
	C_range = 2. ** numpy.arange(-5,15,2)
	gamma_range = 2. ** numpy.arange(3,-15,-2)
	k = [ 'rbf']
	#k = ['linear', 'rbf']
	param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

	# instancia o classificador, gerando probabilidades
	srv = svm.SVC(probability=True)

	# faz a busca
	grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
	grid.fit (X_train, y_train)

	# recupera o melhor modelo
	model = grid.best_estimator_

	# imprime os parametros desse modelo
	print(grid.best_params_)
	return model
def main(datatr, datats):

	# loads data
	print("Loading data...")
	X_train, y_train = load_svmlight_file(datatr)
	X_test, y_test = load_svmlight_file(datats)

	# splits data
	#print "Spliting data..."
	#X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=0.5)

	# fazer a normalizacao dos dados #######
	#scaler = preprocessing.MinMaxScaler()
	#X_train = scaler.fit_transform(X_train)
	#X_test = scaler.transform(X_test)

	# cria um SVM
	#clf =  svm.SVC(kernel='linear', C=1)

	# treina o classificador na base de treinamento
	#print "Training Classifier..."
	#clf.fit(X_train, y_train)

	# mostra o resultado do classificador na base de teste
	#print clf.score(X_test, y_test)

	# predicao do classificador
	#y_pred = clf.predict(X_test)

	# cria a matriz de confusao
	#cm = confusion_matrix(y_test, y_pred)
	#print cm

	# GridSearch retorna o melhor modelo encontrado na busca
	best = GridSearch(X_train, y_train)

	# Treina usando o melhor modelo
	best.fit(X_train, y_train)

	# resultado do treinamento
	print(best.score(X_test, y_test))

	# predicao do classificador
	y_pred = best.predict(X_test)

	# cria a matriz de confusao
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	# probabilidades
	probs = best.predict_proba(X_test)

	print(probs)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: svm.py <tr> <ts>")

	main(sys.argv[1], sys.argv[2])