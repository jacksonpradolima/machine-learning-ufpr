import numpy as np
import random

from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "GPL"
__version__ = "1.0"

# Split my data in 3 set with same size: Train, Test, and Validation.
# Train and Validation are used in the fitness function, after all, the Test dataset is used to evaluate the best individual
features, labels = load_svmlight_file("data/data")

x, X_test, y, y_test = train_test_split(features, labels, test_size=0.3333, random_state=42, stratify=labels)
X_train, X_cv, y_train, y_cv = train_test_split(x, y, test_size=0.5, train_size=0.5, random_state=42, stratify=y)

# Classifier instance
clf = DecisionTreeClassifier()

def run_classifier(_X_train, _X_test, _y_test):
    """
    Execute the classifier
    """
    model = clf.fit(_X_train, y_train)
    predictions = clf.predict(_X_test) 
    score = clf.score(_X_test, _y_test)
    probas = clf.predict_proba(_X_test)

    return predictions, probas, score

def main():
    random.seed(64)

    scores = 0
    for i in range(10):
        predictions, probas, score = run_classifier(X_train, X_test, y_test)
        scores += score
        print("Score: ", score)

    print("\nMédia: {}".format(scores / 10))

if __name__ == "__main__":
    main()
