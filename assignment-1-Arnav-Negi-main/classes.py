import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             multilabel_confusion_matrix, hamming_loss)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

'''
    Class for KNN classification, implementing set_parameters, fit, predict and score methods.
    Vectorization is reduced for this class, as it is the naive implementation. Only the distance metrics are vectorized.
'''


class KNeighborsClassifierNaive:
    def __init__(self, k, metric='euclidean', encoder_type='resnet'):
        self.possible_metrics = {
            'euclidean': self.euclidean,
            'manhattan': self.manhattan,
            'cosine': self.cosine
        }
        self.possible_encoders = {
            'resnet': 0,
            'vit': 1
        }
        self.k = k

        if metric not in self.possible_metrics.keys():
            raise ValueError('Invalid metric')
        if encoder_type not in self.possible_encoders.keys():
            raise ValueError('Invalid encoder type')

        self.metric = self.possible_metrics[metric]
        self.encoder_type = self.possible_encoders[encoder_type]

    # Distance metrics without vectorization. The distance metrics were written with help of copilot
    def euclidean(self, x1, x2):
        return np.sqrt((np.sum((x1 - x2) ** 2, axis=1)))

    def manhattan(self, x1, x2):
        return (np.abs(x1 - x2)).sum(axis=1)

    def cosine(self, x1, x2):
        return 1 - (np.dot(x1, x2.T) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2)))

    def fit(self, X, y):
        self.X = (X[:, 0], X[:, 1])
        self.y = y

    def predict_one(self, x):
        distances = []
        for i in range(len(self.X[self.encoder_type])):
            distances.append((self.metric(x, self.X[self.encoder_type][i]), self.y[i]))

        distances.sort(key=lambda l: l[0])
        label_freq = {}
        for i in range(self.k):
            label_freq[distances[i][1]] = label_freq.get(distances[i][1], 0) + 1
        return max(label_freq, key=label_freq.get)

    def predict(self, X_test):
        X_test = X_test[:, self.encoder_type]
        y_predicted = []
        for i in range(len(X_test)):
            y_predicted.append(self.predict_one(X_test[i]))

        return np.array(y_predicted)

    # returns f1, accuracy, precision, recall. Written with help of copilot
    def score(self, y_predicted, y_true):
        return (f1_score(y_true, y_predicted, average='macro', zero_division=0),
                accuracy_score(y_true, y_predicted),
                precision_score(y_true, y_predicted, average='macro', zero_division=0),
                recall_score(y_true, y_predicted, average='macro', zero_division=0))

    def set_parameters(self, k=None, encoder_type=None, metric=None):
        if k is not None:
            self.k = k
        if encoder_type is not None:
            if encoder_type not in self.possible_encoders.keys():
                raise ValueError('Invalid encoder type')
            self.encoder_type = self.possible_encoders[encoder_type]
        if metric is not None:
            if metric not in self.possible_metrics.keys():
                raise ValueError('Invalid metric')
            self.metric = self.possible_metrics[metric]


'''
    Optimized KNN classification class. This class derives from the naive class.
    The predict method is vectorized using broadcasting. The predict_one method is vectorized using np.vectorize.
    The class gives the same results as the naive class, but it is much faster.
'''


class KNeighborsClassifierOptimized(KNeighborsClassifierNaive):
    def __init__(self, k, metric='euclidean', encoder_type='resnet'):
        super().__init__(k, metric, encoder_type)

    def fit(self, X, y):
        self.X = [None, None]
        self.y = y
        self.X[0] = np.stack(X[:, 0], axis=0).reshape(X[:, 0].shape[0], -1)
        self.X[1] = np.stack(X[:, 1], axis=0).reshape(X[:, 1].shape[0], -1)

    def predict_one(self, x): # written with help of copilot
        # vectorized distance calculation using broadcasting
        distances = self.metric(self.X[self.encoder_type], x.reshape(-1))
        # sorting indices of distances
        sorted_indices = np.argsort(distances)
        # getting the k nearest neighbors
        k_nearest = self.y[sorted_indices[:self.k]]
        # getting the most frequent label
        label_freq = {}
        for lbl in k_nearest:
            label_freq[lbl] = label_freq.get(lbl, 0) + 1
        return max(label_freq, key=label_freq.get)

    def predict(self, X_test):
        # using np.vectorize to vectorize predict_one
        predict_one_v = np.vectorize(self.predict_one)
        return predict_one_v(X_test[:, self.encoder_type])


'''
    Class for Decision Tree classification. It uses the sklearn DecisionTreeClassifier class.
    This is the base abstract class, and is not meant to be used directly. It is used by the two classes below.
'''


class DecisionTreeBase:
    def __init__(self, max_depth=None, criterion='gini', max_features=None):
        if criterion not in ['gini', 'entropy', 'log_loss', None]:
            raise ValueError('Invalid criterion')
        self.tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, max_features=max_features)
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y):
        self.mlb.fit(y)

    def score(self, y_pred, y_true):
        y_pred = self.mlb.transform(y_pred)
        y_true = self.mlb.transform(y_true)

        # return both macro and micro metrics. Written with help of copilot
        return (f1_score(y_true, y_pred, average='micro', zero_division=0),
                f1_score(y_true, y_pred, average='macro', zero_division=0),
                f1_score(y_true, y_pred, average='weighted', zero_division=0),
                accuracy_score(y_true, y_pred),
                1 - hamming_loss(y_true, y_pred),
                precision_score(y_true, y_pred, average='micro', zero_division=0),
                recall_score(y_true, y_pred, average='micro', zero_division=0),
                multilabel_confusion_matrix(y_pred, y_true))


'''
    The derived Decision Tree Classifier for multilabel classification using the powerset formulation.
    It derives from the base class.
'''


class DecisionTreePowerset(DecisionTreeBase):
    def __init__(self, max_depth=None, criterion='gini', max_features=None):
        super().__init__(max_depth, criterion, max_features)

    # The fit and predict method requires the y columns to be a list of labels.
    def fit(self, X, y):
        super().fit(X, y)

        y = y.apply(lambda x: sorted(x)).apply(lambda x: " ".join(x))
        self.tree.fit(X, y)

    def predict(self, X):
        y_pred = self.tree.predict(X)
        y_pred = pd.Series(y_pred).apply(lambda x: x.split(' '))
        return y_pred


'''
    The derived Decision Tree Classifier for multilabel classification using the multiOutput formulation.
    It derives from the base class, and also uses the sklearn MultiOutputClassifier class.
'''


class DecisionTreeMultiOutput(DecisionTreeBase):
    def __init__(self, max_depth=None, criterion='gini', max_features=None):
        super().__init__(max_depth, criterion, max_features)
        self.clf = MultiOutputClassifier(self.tree)

    # The fit and predict method requires the y columns to be a list of labels.
    def fit(self, X, y):
        super().fit(X, y)
        self.clf.fit(X, self.mlb.transform(y))

    def predict(self, X):
        y_pred = self.clf.predict(X)
        y_pred = self.mlb.inverse_transform(y_pred)
        return y_pred
