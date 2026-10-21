# 🧠 Inside-Out Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Optimized-yellow)

## 📌 Overview

This project demonstrates an **"Inside-Out"** approach to Machine Learning. Instead of treating algorithms as "black boxes," core ML models were engineered entirely from scratch using pure Python and NumPy to demonstrate a deep understanding of their mathematical foundations (Probability theory, Linear Algebra, Calculus, and Gradient Descent).

After establishing the mathematical baseline, the pipeline transitions to industry-standard tools (Scikit-Learn) for optimization, scalability, and hyperparameter tuning.

### ✨ Key Highlights matching my :
* **From-Scratch Implementations:** Built core algorithms (Naive Bayes, MLP with Gradient Descent, KNN, SVM logic) without relying on high-level ML libraries.
* **End-to-End Pipeline:** Orchestrated a complete workflow including Exploratory Data Analysis (EDA), missing value imputation, Min-Max scaling, and feature encoding.
* **Optimization & Benchmarking:** Systematically compared the custom "from-scratch" models against their highly optimized Scikit-Learn equivalents to evaluate predictive accuracy and computational efficiency.

## 📂 Project Structure

* `notebooks/` - Contains Jupyter Notebooks detailing the EDA process, feature relationships, and visual evaluations (Confusion Matrices, Decision Boundaries).
* `src/from_scratch/` - The mathematical core. Contains custom Python classes for the ML algorithms.
* `src/preprocessing.py` - Custom data cleaning, scaling, and transformation functions.
* `src/pipeline.py` - The main execution script tying data loading, training, and evaluation together.

## 🚀 The "Inside-Out" Methodology

1. **Understand the Math:** Implement the algorithm using raw NumPy matrices (`@` for dot products, manual `exp` and `log` for Cross-Entropy loss).
2. **Build the Logic:** Write custom `fit()` and `predict()` methods to simulate the API of standard libraries.
3. **Verify & Optimize:** Train a Scikit-Learn model on the exact same preprocessed data and compare the accuracy scores, classification reports, and confusion matrices to ensure the custom logic is mathematically sound.

## 💻 Quick Start

Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/inside-out-ml-pipeline.git
cd inside-out-ml-pipeline
pip install -r requirements.txt
python src/pipeline.py
```