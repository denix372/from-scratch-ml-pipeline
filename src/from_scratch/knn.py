
"""
K-Nearest Neighbors (KNN) Implementation from Scratch.
Demonstrates distance metrics (Euclidean) and majority voting logic.
"""
import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        KNN doesn't learn weights. It simply memorizes the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class label for each instance in X.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Predicts the class for a single instance.
        """
        # 1. Compute Euclidean distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        
        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Return the most common class label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]