from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

class ClassifierUtils:
	def __init__(self, data):
		self.data = data
		self.X = None
		self.y = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

	def loadData(self):
		self.X, self.y = load_svmlight_file(self.data)
		
	def initTrainingTestingSubsets(self, test_size):
	    # We set random_state=1 to guarantee that the split will be always the same for reproducible results drop what will be analysed
	    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=1)
	    
	    #print("Evaluating test size {}% (Train set size = {}, Test set size = {})".format(test_size*100, X_train.shape[0], X_test.shape[0]))    

	def runClassifier(self, model):
		# Train model.
		model = model.fit(self.X_train, self.y_train)

		# Make predictions.
		predictions = model.predict(self.X_test) 

		# Returns the mean accuracy on the given test data and labels.
		score = model.score(self.X_test, self.y_test);	

		#Predict class probabilities of the input samples X_test.
		probas = model.predict_proba(self.X_test)

		return predictions, probas, score