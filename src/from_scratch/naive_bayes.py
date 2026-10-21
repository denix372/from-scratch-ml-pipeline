"""
Gaussian Naive Bayes Implementation from Scratch.
Demonstrates the fundamental probability math behind the Naive Bayes theorem.
"""
import numpy as np

class NaiveBayesFromScratch:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Calculates the mean, variance, and prior probability for each class.
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Initialize arrays for mean, variance, and priors
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            # Prior P(c) = (number of samples in class c) / (total number of samples)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predicts the class for a given set of instances.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts the class for a single instance using log probabilities 
        to prevent numerical underflow.
        """
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            # log(P(c))
            prior = np.log(self.priors[idx])
            # log(P(x|c))
            posterior = np.sum(np.log(self._pdf(idx, x)))

            # P(c|x) ~ log(P(c)) + log(P(x|c))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Probability Density Function (PDF) for a Gaussian distribution.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # Add a small epsilon to variance to prevent division by zero
        eps = 1e-4

        numerator = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        denominator = np.sqrt(2 * np.pi * (var + eps))
        return numerator / denominator