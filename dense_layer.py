import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential

keras = tf.keras

# reading the dataset
df = pd.read_csv("csiro_alt_gmsl_mo_2015_csv_edited.csv")
df['id'] = df.index
time = df.index
series = df.GMSL


def window_dataset(series, window_size, batch_size=10,
                   shuffle_buffer=10):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(16).prefetch(1)
    forecast = model.predict(ds)
    return forecast


split_time = 180
time_train = time[:split_time]
x_train = series[:split_time]
time_test = time[split_time:]
x_test = series[split_time:]

window_size = 5
train_set = window_dataset(x_train, window_size)

network = Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Dense(80, activation="relu", input_shape=[window_size]))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(25, activation="relu"))
model.add(keras.layers.Dense(1))

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
trained = model.fit(train_set, epochs=50)

dense_forecast = model_forecast(
    model,
    series[split_time - window_size:-1],
    window_size)[:, 0]  # first_row:last_row , column_0 ]

# print(dense_forecast)

# plotting

plt.plot(trained.history['mae'], label="MAE")
plt.plot(trained.history['loss'], label="Loss")
plt.legend()
plt.grid(True)
plt.show()

fig, axs = plt.subplots(2)
plt.grid(True)

axs[0].bar(time, series, alpha=1)
axs[0].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Original Data")

axs[1].plot(time_train, x_train, label='Training Data')
axs[1].plot(time_test, x_test, label='Test Data')
axs[1].plot(time_test, dense_forecast, label='Predicted Data')
axs[1].legend(loc='upper left', frameon=True)
axs[1].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Forecasting")

plt.show()

errors = dense_forecast - x_test
abs_errors = np.abs(errors)
mae = abs_errors.mean()
print(mae)
