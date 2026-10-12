# Experiment 203: Multiple Linear Regression and Diagnostics
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import ProbPlot

# a) IPL sold-price regression
df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-3\IPL IMB381IPL2013.csv")
X_features = [
    'AGE', 'COUNTRY', 'PLAYING ROLE', 'T-RUNS', 'WKTS',
    'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL',
    'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B',
    'SIXERS', 'RUNS-C', 'AVE-BL', 'ECON', 'SR-BL'
]
categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']
encoded_df = pd.get_dummies(
    df[X_features], columns=categorical_features, drop_first=True, dtype=int
)
X = sm.add_constant(encoded_df)
y = df['SOLD PRICE']
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.8, random_state=42
)
model = sm.OLS(train_y, train_X).fit()
print(model.summary())

# b) Normal P-P plot of standardized residuals
y_pred = model.predict(test_X)
residuals = test_y - y_pred
standardized_residuals = (residuals - residuals.mean()) / residuals.std()
sm.ProbPlot(standardized_residuals).ppplot(line='45')
plt.title("Normal P-P Plot of Residuals")
plt.xlabel("Theoretical Cumulative Probabilities")
plt.ylabel("Observed Cumulative Probabilities")
plt.grid(True)
plt.show()

# c) House-price MLR correlation matrix
df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-3\house_price_mlr_100_samples.csv")
X_features = ['Area', 'Bedrooms', 'BuiltUp', 'Age']
X = sm.add_constant(df[X_features])
y = df['Price']
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.8, random_state=42
)
model_2 = sm.OLS(train_y, train_X).fit()

corr_matrix = df[X_features].corr()
print(corr_matrix.round(3))
plt.figure(figsize=(6, 5))
sn.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', square=True)
plt.title("Correlation Matrix")
plt.show()

# d) Residual P-P plot for house-price model
resid = model_2.resid
probplot = ProbPlot(resid)
plt.figure(figsize=(10, 8))
probplot.ppplot(line="45")
plt.title("P-P Plot")
plt.xlabel("Theoretical probabilities")
plt.ylabel("Sample probabilities")
plt.show()
