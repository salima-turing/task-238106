import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = pd.Series(iris.target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def test_select_k_best_features_various_k(self):
        best_accuracy = 0
        best_k = 0

        for k in range(1, len(self.X.columns) + 1):
            feature_selector = SelectKBest(score_func=f_classif, k=k)
            feature_selector.fit(self.X_train, self.y_train)
            X_train_selected = feature_selector.transform(self.X_train)
            X_test_selected = feature_selector.transform(self.X_test)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_selected, self.y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(self.y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        acceptance_threshold = 0.8
        self.assertGreaterEqual(best_accuracy, acceptance_threshold,
                                f"Best accuracy {best_accuracy} is below the acceptance threshold {acceptance_threshold}")
        print(f"Best k: {best_k}, Best Accuracy: {best_accuracy}")


if __name__ == '__main__':
    unittest.main()
