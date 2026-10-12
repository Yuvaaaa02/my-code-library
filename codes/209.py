# Experiment 209: ARIMA Demand Forecasting
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-9\store.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

y = df['demand']

# a) Fit ARIMA model and forecast next 10 values
model = ARIMA(y, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

print("Forecasted Demand:")
print(forecast)

plt.figure(figsize=(10, 5))
plt.plot(y, label='Actual Demand')
plt.plot(forecast, label='Forecasted Demand', color='red')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('ARIMA Demand Forecasting')
plt.legend()
plt.show()

# b) Display model parameters and evaluation information
print("ARIMA Model Parameters:")
print(model_fit.params)
print("Log Likelihood:", model_fit.llf)
print("AIC:", model_fit.aic)
print("BIC:", model_fit.bic)
