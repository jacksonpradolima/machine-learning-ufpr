#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix 
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB


import os


def main(data):

	# loads data
	print "Loading data..."
	X_data, y_data = load_svmlight_file(data)
    # splits data
	print "Spliting data..."
	X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=0.5)

	X_train = X_train.toarray()
	# cria o classificador
	
	### Note that GaussianNB wont fit in this kind of sparse data (a lot of zero features) 
	#gnb  = GaussianNB() 
	gnb  = BernoulliNB() 

	print 'Fitting NB'
	gnb.fit(X_train, y_train)

	# predicao do classificador
	print 'Predicting...'
	y_pred = gnb.predict(X_test) 

	# mostra o resultado do classificador na base de teste
	print gnb.score(X_test, y_test)

	# cria a matriz de confusao
	cm = confusion_matrix(y_test, y_pred)
	print cm


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Use: svm.py <data>")

	main(sys.argv[1])


