class ClassifierResult:
	def __init__(self, X_train, X_test, y_train, y_test, predictions, probas, score, weight_train, weight_test):
		
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		self.predictions = predictions
		self.probas = probas
		self.score = score

		self.weight_train = weight_train
		self.weight_test = weight_test