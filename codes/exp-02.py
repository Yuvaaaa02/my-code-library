# ============================================================
# CM354 MACHINE LEARNING LAB
# EXPERIMENT 2
# SIMPLE LINEAR REGRESSION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from scipy import stats


# ============================================================
# PROBLEM 1
# SIMPLE LINEAR REGRESSION USING RANDOM DATA
# LEAST SQUARE METHOD
# ============================================================

print("\n==============================================")
print("PROBLEM 1 : RANDOM DATASET")
print("==============================================")

# Generate random data
np.random.seed(42)

X = np.arange(1, 21)

Y = 3 * X + 5 + np.random.normal(0, 5, 20)

# Calculate means
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Calculate slope using Least Square Method
b1 = np.sum(
    (X - X_mean) * (Y - Y_mean)
) / np.sum(
    (X - X_mean) ** 2
)

# Calculate intercept
b0 = Y_mean - b1 * X_mean

# Prediction
Y_pred = b0 + b1 * X

print("Slope =", b1)
print("Intercept =", b0)

print(
    "\nRegression Equation:"
)
print(
    f"Y = {b0:.2f} + {b1:.2f}X"
)

# R2
r2 = r2_score(Y, Y_pred)

print("R2 =", r2)

# Plot
plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    Y,
    label="Data Points"
)

plt.plot(
    X,
    Y_pred,
    label="Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "Problem 1 - Least Square Regression"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# PROBLEM 2
# MBA SALARY DATASET
# ============================================================

print("\n==============================================")
print("PROBLEM 2 : MBA SALARY DATASET")
print("==============================================")


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv("MBA Salary.csv")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 2. Select X and Y
# ------------------------------------------------------------

X = df[["Percentage in Grade 10"]]

y = df["Salary"]


# ------------------------------------------------------------
# 3. Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y
)

plt.xlabel(
    "Percentage in Grade 10"
)

plt.ylabel(
    "Salary"
)

plt.title(
    "Grade 10 Percentage vs Salary"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 4. Build SLR Model
# ------------------------------------------------------------

model = LinearRegression()

model.fit(
    X,
    y
)


# ------------------------------------------------------------
# 5. Slope and Intercept
# ------------------------------------------------------------

b0 = model.intercept_

b1 = model.coef_[0]

print("\nIntercept =", b0)

print("Slope =", b1)

print(
    "\nRegression Equation:"
)

print(
    f"Salary = {b0:.2f} + "
    f"{b1:.2f} × Percentage"
)


# ------------------------------------------------------------
# 6. Prediction
# ------------------------------------------------------------

y_pred = model.predict(X)

df["Predicted Salary"] = y_pred

df["Residual"] = (
    df["Salary"] -
    df["Predicted Salary"]
)


# ------------------------------------------------------------
# 7. R2
# ------------------------------------------------------------

r2 = r2_score(
    y,
    y_pred
)

print(
    "\nR2 Score =",
    r2
)


# ------------------------------------------------------------
# 8. Regression Line
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y,
    label="Actual"
)

plt.plot(
    X,
    y_pred,
    label="Regression Line"
)

plt.xlabel(
    "Percentage in Grade 10"
)

plt.ylabel(
    "Salary"
)

plt.title(
    "MBA Salary - Simple Linear Regression"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 9. Residual Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    y_pred,
    df["Residual"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Salary"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - MBA Salary"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 10. Statsmodels Regression Diagnosis
# ------------------------------------------------------------

X_sm = sm.add_constant(
    df["Percentage in Grade 10"]
)

ols_model = sm.OLS(
    df["Salary"],
    X_sm
).fit()

print(
    "\n========== REGRESSION SUMMARY =========="
)

print(
    ols_model.summary()
)


# ------------------------------------------------------------
# 11. P-P Plot
# ------------------------------------------------------------

residuals = ols_model.resid

z = (
    residuals -
    residuals.mean()
) / residuals.std(ddof=1)

z_sorted = np.sort(z)

theoretical = stats.norm.cdf(
    z_sorted
)

n = len(z_sorted)

observed = (
    np.arange(1, n + 1) - 0.5
) / n

plt.figure(figsize=(7, 5))

plt.scatter(
    theoretical,
    observed
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "Theoretical Probability"
)

plt.ylabel(
    "Observed Probability"
)

plt.title(
    "P-P Plot - MBA Salary"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 12. Z-SCORE OUTLIERS
# ------------------------------------------------------------

df["Percentage_Z"] = stats.zscore(
    df["Percentage in Grade 10"]
)

df["Salary_Z"] = stats.zscore(
    df["Salary"]
)

z_outliers = df[
    (abs(df["Percentage_Z"]) > 3) |
    (abs(df["Salary_Z"]) > 3)
]

print(
    "\n========== Z-SCORE OUTLIERS =========="
)

print(
    z_outliers[
        [
            "S. No.",
            "Percentage in Grade 10",
            "Salary",
            "Percentage_Z",
            "Salary_Z"
        ]
    ]
)


# ------------------------------------------------------------
# 13. COOK'S DISTANCE
# ------------------------------------------------------------

influence = ols_model.get_influence()

cooks_d = (
    influence.cooks_distance[0]
)

df["Cooks_Distance"] = cooks_d

threshold = 4 / len(df)

print(
    "\nCook's Distance Threshold =",
    threshold
)

cook_outliers = df[
    df["Cooks_Distance"] > threshold
]

print(
    "\n========== COOK'S DISTANCE OUTLIERS =========="
)

print(
    cook_outliers[
        [
            "S. No.",
            "Percentage in Grade 10",
            "Salary",
            "Cooks_Distance"
        ]
    ]
)


# ------------------------------------------------------------
# 14. Cook's Distance Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.stem(
    range(1, len(df) + 1),
    cooks_d
)

plt.axhline(
    threshold,
    linestyle="--",
    label="4/n"
)

plt.xlabel(
    "Observation"
)

plt.ylabel(
    "Cook's Distance"
)

plt.title(
    "Cook's Distance - MBA Salary"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 15. Predict Salary for New Student
# ------------------------------------------------------------

new_percentage = [[75]]

predicted_salary = model.predict(
    new_percentage
)

print(
    "\nPredicted Salary for 75% =",
    predicted_salary[0]
)


# ------------------------------------------------------------
# 16. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

test_model = LinearRegression()

test_model.fit(
    X_train,
    y_train
)

y_test_pred = test_model.predict(
    X_test
)


# ------------------------------------------------------------
# 17. Accuracy
# ------------------------------------------------------------

r2_test = r2_score(
    y_test,
    y_test_pred
)

mae = mean_absolute_error(
    y_test,
    y_test_pred
)

mse = mean_squared_error(
    y_test,
    y_test_pred
)

rmse = np.sqrt(mse)

print(
    "\n========== MBA TEST PERFORMANCE =========="
)

print(
    "R2   =", r2_test
)

print(
    "MAE  =", mae
)

print(
    "MSE  =", mse
)

print(
    "RMSE =", rmse
)


# ============================================================
# PROBLEM 3
# COUNTRY DATASET
# ============================================================

print("\n==============================================")
print("PROBLEM 3 : COUNTRY DATASET")
print("==============================================")


# ------------------------------------------------------------
# 1. Load country dataset
# ------------------------------------------------------------

country = pd.read_csv(
    "country.csv"
)

print("\nFirst 5 records:")
print(
    country.head()
)

print("\nColumns:")
print(
    country.columns
)

print("\nMissing Values:")
print(
    country.isnull().sum()
)


# ------------------------------------------------------------
# 2. Select X and Y
# ------------------------------------------------------------
# Change these column names if your CSV has different names.

X = country[
    ["Gini Index"]
]

y = country[
    "Corruption Perception Index"
]


# ------------------------------------------------------------
# 3. Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y
)

plt.xlabel(
    "Gini Index"
)

plt.ylabel(
    "Corruption Perception Index"
)

plt.title(
    "Gini Index vs Corruption Perception Index"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 4. Build SLR Model
# ------------------------------------------------------------

country_model = LinearRegression()

country_model.fit(
    X,
    y
)


# ------------------------------------------------------------
# 5. Slope and Intercept
# ------------------------------------------------------------

b0 = country_model.intercept_

b1 = country_model.coef_[0]

print(
    "\nIntercept =",
    b0
)

print(
    "Slope =",
    b1
)

print(
    "\nRegression Equation:"
)

print(
    f"CPI = {b0:.4f} + "
    f"({b1:.4f}) × Gini"
)


# ------------------------------------------------------------
# 6. Prediction
# ------------------------------------------------------------

country_pred = country_model.predict(
    X
)

country["Predicted CPI"] = (
    country_pred
)

country["Residual"] = (
    country["Corruption Perception Index"]
    - country["Predicted CPI"]
)


# ------------------------------------------------------------
# 7. R2
# ------------------------------------------------------------

r2 = r2_score(
    y,
    country_pred
)

print(
    "\nR2 Score =",
    r2
)


# ------------------------------------------------------------
# 8. Regression Line
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y,
    label="Actual"
)

plt.plot(
    X,
    country_pred,
    label="Regression Line"
)

plt.xlabel(
    "Gini Index"
)

plt.ylabel(
    "Corruption Perception Index"
)

plt.title(
    "Country Dataset - SLR"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 9. Residual Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    country_pred,
    country["Residual"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted CPI"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Country Dataset"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 10. Statsmodels Diagnosis
# ------------------------------------------------------------

X_sm = sm.add_constant(
    country["Gini Index"]
)

country_ols = sm.OLS(
    country[
        "Corruption Perception Index"
    ],
    X_sm
).fit()

print(
    "\n========== COUNTRY REGRESSION SUMMARY =========="
)

print(
    country_ols.summary()
)


# ------------------------------------------------------------
# 11. P-P Plot
# ------------------------------------------------------------

residuals = country_ols.resid

z = (
    residuals -
    residuals.mean()
) / residuals.std(ddof=1)

z_sorted = np.sort(z)

theoretical = stats.norm.cdf(
    z_sorted
)

n = len(z_sorted)

observed = (
    np.arange(1, n + 1) - 0.5
) / n

plt.figure(figsize=(7, 5))

plt.scatter(
    theoretical,
    observed
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "Theoretical Probability"
)

plt.ylabel(
    "Observed Probability"
)

plt.title(
    "P-P Plot - Country Dataset"
)

plt.grid()

plt.show()


# ------------------------------------------------------------
# 12. Z-SCORE OUTLIERS
# ------------------------------------------------------------

country["Gini_Z"] = stats.zscore(
    country["Gini Index"]
)

country["CPI_Z"] = stats.zscore(
    country[
        "Corruption Perception Index"
    ]
)

z_outliers = country[
    (abs(country["Gini_Z"]) > 3) |
    (abs(country["CPI_Z"]) > 3)
]

print(
    "\n========== COUNTRY Z-SCORE OUTLIERS =========="
)

print(
    z_outliers
)


# ------------------------------------------------------------
# 13. COOK'S DISTANCE
# ------------------------------------------------------------

influence = (
    country_ols.get_influence()
)

cooks_d = (
    influence.cooks_distance[0]
)

country["Cooks_Distance"] = cooks_d

threshold = 4 / len(country)

print(
    "\nCountry Cook's Distance Threshold =",
    threshold
)

cook_outliers = country[
    country["Cooks_Distance"] > threshold
]

print(
    "\n========== COUNTRY COOK'S OUTLIERS =========="
)

print(
    cook_outliers[
        [
            "Gini Index",
            "Corruption Perception Index",
            "Cooks_Distance"
        ]
    ]
)


# ------------------------------------------------------------
# 14. Cook's Distance Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.stem(
    range(1, len(country) + 1),
    cooks_d
)

plt.axhline(
    threshold,
    linestyle="--",
    label="4/n"
)

plt.xlabel(
    "Observation"
)

plt.ylabel(
    "Cook's Distance"
)

plt.title(
    "Cook's Distance - Country Dataset"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 15. Predict CPI for New Gini Value
# ------------------------------------------------------------

new_gini = [[35]]

predicted_cpi = (
    country_model.predict(
        new_gini
    )
)

print(
    "\nPredicted CPI for Gini = 35:"
)

print(
    predicted_cpi[0]
)


# ------------------------------------------------------------
# 16. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

country_test_model = (
    LinearRegression()
)

country_test_model.fit(
    X_train,
    y_train
)

y_test_pred = (
    country_test_model.predict(
        X_test
    )
)


# ------------------------------------------------------------
# 17. Accuracy
# ------------------------------------------------------------

r2_test = r2_score(
    y_test,
    y_test_pred
)

mae = mean_absolute_error(
    y_test,
    y_test_pred
)

mse = mean_squared_error(
    y_test,
    y_test_pred
)

rmse = np.sqrt(mse)

print(
    "\n========== COUNTRY TEST PERFORMANCE =========="
)

print(
    "R2   =", r2_test
)

print(
    "MAE  =", mae
)

print(
    "MSE  =", mse
)

print(
    "RMSE =", rmse
)


# ============================================================
# END
# ============================================================

print("\n==============================================")
print("EXPERIMENT 2 COMPLETED")
print("==============================================")
