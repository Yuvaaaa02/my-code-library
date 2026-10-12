# Experiment 202: Simple Linear Regression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# a) Demonstration dataset
data = {
    'Study Hours': [1, 2, 3, 4],
    'Scores': [40, 50, 60, 70]
}
df = pd.DataFrame(data)

X = sm.add_constant(df[['Study Hours']])
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100
)
lm = sm.OLS(y_train, X_train).fit()

prediction = lm.predict(pd.DataFrame({'const': [1], 'Study Hours': [3.5]}))
print("Predicted Score:", prediction.iloc[0])

predicted = lm.predict(X)
plt.figure(figsize=(6, 4))
plt.scatter(df['Study Hours'], df['Scores'], color='blue', s=80, label='Data Points')
plt.plot(df['Study Hours'], predicted, color='red', label='Regression Line')
plt.title("Simple Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

# b) Predict salary from Grade 10 percentage
df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-2\MBA Salary.csv")
X = sm.add_constant(df[['Percentage in Grade 10']])
y = df['Salary']
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.8, random_state=100
)
lm = sm.OLS(train_y, train_X).fit()
print(lm.params)

new_data = pd.DataFrame({'const': [1], 'Percentage in Grade 10': [60]})
prediction = lm.predict(new_data)
print(prediction)
print("prediction:", lm.predict(pd.DataFrame({'const': [1], 'Percentage in Grade 10': [60]})).iloc[0])

y_pred = lm.predict(test_X)
print("R2 Score:", round(r2_score(test_y, y_pred), 3))
print("MAE:", round(mean_absolute_error(test_y, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(test_y, y_pred)), 2))
