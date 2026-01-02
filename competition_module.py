
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class NaiveBayesLearner:
    """Naive Bayes classifier wrapper"""
    def __init__(self):
        self.model = MultinomialNB()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)


class DecisionTreeLearner:
    """Decision Tree classifier wrapper"""
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)


class NeuralNetLearner:
    """Neural Network classifier wrapper"""
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=200):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)


class KNearestNeighborsLearner:
    """K-Nearest Neighbors classifier wrapper"""
    def __init__(self, k=5):
        self.model = KNeighborsClassifier(n_neighbors=k)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)


class SVMLearner:
    """Support Vector Machine classifier wrapper"""
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel, random_state=42)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

from sklearn.ensemble import RandomForestClassifier

class RandomForestLearner:
    """Random Forest classifier wrapper"""
    def __init__(self, n_estimators=100, max_depth=None, max_features='auto'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
from sklearn.linear_model import LogisticRegression

class LogisticRegressionLearner:
    """Logistic Regression classifier wrapper"""
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs'):
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

