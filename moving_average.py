import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


# reading the dataset
df = pd.read_csv("csiro_alt_gmsl_mo_2015_csv_edited.csv")
df['id'] = df.index
time = df.index
series = df.GMSL

# plot_series(time, series, label="Training Set")

split_time = 180
time_train = time[:split_time]
x_train = series[:split_time]
time_test = time[split_time:]
x_test = series[split_time:]

moving_avg = moving_average_forecast(series, 5)[split_time - 5:]
print(moving_avg)

fig, axs = plt.subplots(2)
plt.grid(True)

axs[0].bar(time, series, alpha=1)
axs[0].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Original Data")

axs[1].plot(time_train, x_train, label='Training Data')
axs[1].plot(time_test, x_test, label='Test Data')
axs[1].plot(time_test, moving_avg, label='Predicted Data')
axs[1].legend(loc='upper left', frameon=True)
axs[1].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Forecasting")

plt.show()

errors = moving_avg - x_test
abs_errors = np.abs(errors)
mae = abs_errors.mean()
print("Mean Absoltute Error", mae)
