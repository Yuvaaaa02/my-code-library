import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Set Date as index
df.set_index("Date", inplace=True)

# Select time-series column
data = df["Value"]

# Display data
print(data)

# Plot original time series
plt.plot(data)
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Data")
plt.show()

# Create ARIMA model
model = ARIMA(data, order=(1, 1, 1))

# Train model
model_fit = model.fit()

# Forecast next 5 values
forecast = model_fit.forecast(steps=5)

# Display forecast
print("Forecasted Values:")
print(forecast)

# Plot forecast
plt.plot(data, label="Actual")
plt.plot(forecast, label="Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Forecast")
plt.legend()
plt.show()
