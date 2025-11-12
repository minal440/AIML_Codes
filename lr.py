# LinearRegression.py
# -----------------------------------------------
# LINEAR REGRESSION — PREDICTING CONTINUOUS VALUES
# -----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
import seaborn as sns

# Step 1: Generate synthetic data (y = 3x + noise)
X, y = make_regression(n_samples=100, n_features=1, noise=15, coef=False, random_state=42)
y = 3 * X.squeeze() + np.random.randn(100) * 10  # add some extra noise

# Step 2: Visualize raw data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.6, edgecolors='k')
plt.title("Raw Data - Linear Relationship")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.show()

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Linear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")
print("\nModel Coefficients:")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Plot Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Step 8: Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.title("Residual Distribution")
plt.xlabel("Residual Error")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color='orange', edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()
