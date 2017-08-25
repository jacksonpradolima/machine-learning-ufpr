#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_svmlight_file
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def main(data):
    plot_step = .02  # step size in the mesh

    names = ["Decision Tree", "Naive Bayes"]

    classifiers = [DecisionTreeClassifier(max_depth=5), GaussianNB()]

    # preprocess dataset, split into training and test part
    X, y = load_svmlight_file(data)
    
    figure = plt.figure(figsize=(27, 9))
    i = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # just plot the dataset first
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax = plt.subplot(1, len(classifiers) + 1, i)
    # #if ds_cnt == 0:
    # ax.set_title("Input data")
    
    # # Plot the training points
    # # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # # # and testing points
    # # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each        
            Z = clf.predict_proba(X_test)[

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        #if ds_cnt == 0:
        ax.set_title(name)
        #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
         #       size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":  
    if len(sys.argv) != 2:
        sys.exit("Use: svm.py <data>")

    main(sys.argv[1])