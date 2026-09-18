import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Load dataset
df = pd.read_csv("Dataset/HDI.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Input features
X = df[
    [
        "Life expectancy",
        "Mean years of schooling",
        "Expected years of schooling",
        "Gross national income (GNI) per capita"
    ]
]

# Target
y = df["Human Development Index (HDI)"]

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# Remove missing values
data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data[
    [
        "Life expectancy",
        "Mean years of schooling",
        "Expected years of schooling",
        "Gross national income (GNI) per capita"
    ]
]

y = data["Human Development Index (HDI)"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation")
print("-----------------------")
print("R2 Score :", r2)
print("MAE      :", mae)
print("MSE      :", mse)
print("RMSE     :", rmse)

# Save model
joblib.dump(model, "wellbeing_model.pkl")

print("\nModel saved successfully!")
print("File: wellbeing_model.pkl")