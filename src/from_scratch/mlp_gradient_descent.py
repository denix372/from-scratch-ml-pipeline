"""
Multi-Layer Perceptron (MLP) Implementation from Scratch.
Demonstrates Forward Pass, Backpropagation (Chain Rule), and Gradient Descent
for a basic neural network with one hidden layer.
"""
import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases randomly (scaled down for stability)
        # W1: (input_dim, hidden_dim), b1: (1, hidden_dim)
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        
        # W2: (hidden_dim, output_dim), b2: (1, output_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def _relu(self, Z):
        """Rectified Linear Unit (ReLU) activation function."""
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """Derivative of ReLU for backpropagation."""
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        """Softmax activation for multi-class probabilities."""
        # Shift Z for numerical stability (prevent exponential overflow)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _one_hot_encode(self, y, num_classes):
        """Converts integer labels to one-hot encoded matrix."""
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):
        """
        Trains the neural network using Gradient Descent.
        """
        n_samples = X.shape[0]
        output_dim = self.W2.shape[1]
        
        # Encode true labels
        Y_one_hot = self._one_hot_encode(y, output_dim)

        for epoch in range(self.epochs):
            # FORWARD PASS,  Hidden layer
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self._relu(Z1)
            
            # Output layer
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self._softmax(Z2)

            # --- BACKWARD PASS (Gradient Descent & Chain Rule) ---
            # Error at output layer (Predictions - True Labels)
            dZ2 = A2 - Y_one_hot
            dW2 = np.dot(A1.T, dZ2) / n_samples
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

            # Error propagated to hidden layer
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._relu_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) / n_samples
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

            # upgrade weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            # Print loss every 100 epochs for tracking
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                # Cross-Entropy Loss
                loss = -np.mean(np.sum(Y_one_hot * np.log(A2 + 1e-9), axis=1))
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predicts the class label for new data.
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._softmax(Z2)
        
        # Return the class with the highest probability
        return np.argmax(A2, axis=1)