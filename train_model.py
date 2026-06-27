# ==========================================================
# Project Title : A Comprehensive Measure of Well-Being
# File          : train_model.py
# Description   : Machine Learning Model Training
# ==========================================================

# ==========================
# Import Required Libraries
# ==========================
import pandas as pd

# Importing the dataset
Development = pd.read_csv("dataset/HDI.csv")

# Listing the first five rows
print(Development.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

print("="*60)
print("A Comprehensive Measure of Well-Being")
print("Required Libraries Imported Successfully")
print("="*60)
# ==========================================================
# Epic 3 : Dataset Download and Understanding
# Story 1 : Reading the Dataset
# ==========================================================

print("\nLoading HDI Dataset...")

# Read Dataset
data = pd.read_csv("dataset/HDI.csv")

print("Dataset Loaded Successfully")

# Display First 5 Rows
print("\nFirst 5 Rows of Dataset")
print(data.head())
# ==========================================
# Dataset Information
# ==========================================
# Fill Missing Values
# ==========================================

print("\nFilling Missing Values...")

numeric_columns = data.select_dtypes(include=['number']).columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].mean()
)

print("Missing Values Filled Successfully")
# ==========================================
# ==========================================
# Encode Country
# ==========================================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["Country"] = encoder.fit_transform(data["Country"])

print("Country Column Encoded Successfully")

print("\nDataset Shape")
print(data.shape)

print("\nColumn Names")
print(data.columns)

print("\nMissing Values")
print(data.isnull().sum())

# Display Dataset Shape
print("\nDataset Shape")
print(data.shape)
import pandas as pd

data = pd.read_csv("dataset/HDI.csv")

print(data.columns)
print(data.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Encode Country
encoder = LabelEncoder()
data["Country"] = encoder.fit_transform(data["Country"])

# Features
X = data[
    [
        "Country",
        "Life expectancy",
        "Expected years of schooling",
        "Mean years of schooling",
        "Gross national income (GNI) per capita"
    ]
]

# Target
y = data["Human Development Index (HDI)"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Training Completed")
import os
import joblib

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/wellbeing_model.pkl")

print("Model Saved Successfully")