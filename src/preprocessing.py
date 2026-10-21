"""
Data Preprocessing Module.
Handles data loading, exploratory data analysis (EDA), imputation,
encoding, and feature scaling.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_titanic():
    """Loads Titanic dataset and performs basic preprocessing for ML."""
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    
    print("--- Exploratory Data Analysis ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values:\n{df.isna().sum()[df.isna().sum() > 0]}\n")
    
    # Simple imputation and encoding
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True, dtype=int)
    
    # Drop columns that need advanced NLP or aren't immediately useful for this demo
    df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'])
    df = df.dropna() # Drop remaining nan just in case
    
    X = df.drop(columns=['survived']).values
    y = df['survived'].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def scale_data(X_train, X_test):
    """
    Handles feature scaling.
    Crucial ML principle: fit the scaler ONLY on training data to prevent data leakage.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled