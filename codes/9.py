import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("sales.csv")

# Create time series
df["Month"] = pd.to_datetime(df["Month"], format="%m")

# Set Month as index
df.set_index("Month", inplace=True)

# Select Sale Quantity
sales = df["Sale Quantity"]

# Plot the time series
plt.plot(sales)
plt.xlabel("Month")
plt.ylabel("Sale Quantity")
plt.title("Sales Time Series")
plt.show()

# Create ARIMA model
model = ARIMA(sales, order=(1, 1, 1))

# Train the model
model_fit = model.fit()

# Forecast next 6 months
forecast = model_fit.forecast(steps=6)

# Display forecast
print("Forecasted Sale Quantity:")
print(forecast)

# Plot actual and forecast
plt.plot(sales, label="Actual Sales")
plt.plot(forecast, label="Forecast")
plt.xlabel("Month")
plt.ylabel("Sale Quantity")
plt.title("Sales Forecast")
plt.legend()
plt.show()
