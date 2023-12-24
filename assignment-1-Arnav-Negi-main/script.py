from classes import KNeighborsClassifierOptimized
import numpy as np
import sys

# Load data
dataset = np.load('data.npy', allow_pickle=True)[:, 1:4]
X, y = dataset[:, :-1], dataset[:, -1]

knn_model = KNeighborsClassifierOptimized(k=5, metric='manhattan', encoder_type='vit')
knn_model.fit(X, y)

test_data = np.load(sys.argv[1], allow_pickle=True)[:, 1:4]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

f1, accuracy, precision, recall = knn_model.score(knn_model.predict(X_test), y_test)
print(f'F1: {f1}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}')
