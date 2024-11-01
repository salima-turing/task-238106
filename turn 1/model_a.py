import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestFeatureSelection(unittest.TestCase):

	def setUp(self):
		# Load the Iris dataset
		iris = load_iris()
		self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
		self.y = pd.Series(iris.target)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

	def test_select_k_best_features(self):
		# Define the feature selection method (SelectKBest with f_classif)
		num_features = 2
		feature_selector = SelectKBest(score_func=f_classif, k=num_features)

		# Fit the feature selector on training data
		feature_selector.fit(self.X_train, self.y_train)

		# Transform the training and test data
		X_train_selected = feature_selector.transform(self.X_train)
		X_test_selected = feature_selector.transform(self.X_test)

		# Example model using selected features
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression(max_iter=200)
		model.fit(X_train_selected, self.y_train)
		y_pred = model.predict(X_test_selected)

		# Accuracy assessment
		accuracy = accuracy_score(self.y_test, y_pred)

		# Set an acceptance threshold for accuracy
		acceptance_threshold = 0.9

		self.assertGreaterEqual(accuracy, acceptance_threshold, f"Feature selection accuracy {accuracy} is below the acceptance threshold {acceptance_threshold}")

if __name__ == '__main__':
	unittest.main()
