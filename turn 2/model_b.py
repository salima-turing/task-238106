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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)


def test_select_k_best_features_vary_k(self):
    # Define a list of different values of k to test
    k_values = [1, 2, 3, 4]
    best_accuracy = 0
    best_k = None

    for k in k_values:
        # Define the feature selection method (SelectKBest with f_classif)
        feature_selector = SelectKBest(score_func=f_classif, k=k)

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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # Assert that the best_accuracy is above the acceptance threshold
    acceptance_threshold = 0.8
    self.assertGreaterEqual(best_accuracy, acceptance_threshold,
                            f"Best accuracy {best_accuracy} is below the acceptance threshold {acceptance_threshold} for any k value.")
    print(f"Best k: {best_k}, Best Accuracy: {best_accuracy}")


if __name__ == '__main__':
    unittest.main()
