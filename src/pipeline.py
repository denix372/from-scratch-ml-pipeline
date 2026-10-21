"""
End-to-End Machine Learning Pipeline: "Inside-Out" Approach
Demonstrates EDA, Preprocessing, from-scratch algorithmic implementations (Naive Bayes, MLP, KNN),
and systematic hyperparameter tuning using Scikit-Learn.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_and_preprocess_titanic, scale_data

from from_scratch.naive_bayes import NaiveBayesFromScratch
from from_scratch.mlp_gradient_descent import MultiLayerPerceptron
from from_scratch.knn import KNNFromScratch

if __name__ == "__main__":
    print("Initializing Inside-Out ML Pipeline...\n")

    # 1. Data Loading, EDA & Splitting
    X_train, X_test, y_train, y_test = load_and_preprocess_titanic()

    # 2. Scaling (Fit strictly on train data to avoid leakage)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # 3. Model Comparisons
    # We compare our from-scratch implementations directly against Scikit-Learn's optimized versions
    models = {
        "Naive Bayes": {
            "custom": NaiveBayesFromScratch(),
            "sklearn": GaussianNB()
        },
        "K-Nearest Neighbors": {
            "custom": KNNFromScratch(k=5),
            "sklearn": KNeighborsClassifier(n_neighbors=5)
        },
        "Multi-Layer Perceptron": {
            "custom": MultiLayerPerceptron(input_dim=X_train_scaled.shape[1], hidden_dim=8, output_dim=2, epochs=500),
            "sklearn": MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
        }
    }

    for name, implementations in models.items():
        print(f"========== {name} ==========")
        
        # Custom Implementation
        custom_model = implementations["custom"]
        custom_model.fit(X_train_scaled, y_train)
        custom_preds = custom_model.predict(X_test_scaled)
        custom_acc = accuracy_score(y_test, custom_preds)
        print(f"Custom Accuracy: {custom_acc * 100:.2f}%")
        
        # Scikit-Learn Implementation
        sklearn_model = implementations["sklearn"]
        sklearn_model.fit(X_train_scaled, y_train)
        sklearn_preds = sklearn_model.predict(X_test_scaled)
        sklearn_acc = accuracy_score(y_test, sklearn_preds)
        print(f"Sklearn Accuracy:  {sklearn_acc * 100:.2f}%\n")

    # 4. Hyperparameter Tuning
    print("========== Systematic Hyperparameter Tuning ==========")
    print("Optimizing Scikit-Learn MLPClassifier via GridSearchCV...")
    
    param_grid = {
        'hidden_layer_sizes': [(8,), (16,), (8, 4)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
    
    # use n_jobs=-1 to use all CPU cores
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    tuned_preds = best_model.predict(X_test_scaled)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Tuned Model Accuracy: {tuned_acc * 100:.2f}%")