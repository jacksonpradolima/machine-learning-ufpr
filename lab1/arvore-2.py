#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys

from sklearn import tree
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



def main(data):

	# loads data

	print("Loading data...")
	#X_train, y_train = load_svmlight_file(train)
	#X_test, y_test = load_svmlight_file(test)

	X_data, y_data = load_svmlight_file(data)
    # splits data
	#print "Spliting data..."
	X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5)

	# cria uma DT
	clf  = tree.DecisionTreeClassifier() 

	print('Fitting DT')
	clf.fit(X_train, y_train)

	# predicao do classificador
	print('Predicting...')
	y_pred = clf.predict(X_test) 

	# mostra o resultado do classificador na base de teste
	print(clf.score(X_test, y_test))

	# cria a matriz de confusao
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	#with open("tree.dot", 'w') as f:
	#	f= tree.export_graphviz(clf, out_file=f)
		
		#dot -Tpdf tree.dot -o tree.pdf
		
	### Probabilities
	probs = clf.predict_proba(X_test)
	print(probs)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Use: svm.py <data>")

	main(sys.argv[1])


