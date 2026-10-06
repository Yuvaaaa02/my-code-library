import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("store.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

sales = df["Sales"]

plt.plot(sales)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Time Series")
plt.show()

model = ARIMA(sales, order=(1, 1, 1))

model_fit = model.fit()

forecast = model_fit.forecast(steps=10)

print("Forecast:")
print(forecast)

plt.plot(sales, label="Actual")
plt.plot(
    forecast.index,
    forecast,
    label="Forecast"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()