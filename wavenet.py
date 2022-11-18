import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras


def seq2seq_window_dataset(series, window_size, batch_size=10,
                           shuffle_buffer=10):
    series = tf.expand_dims(series, axis=-1)  # expand dimention of matrix, axis=-1 means last axis(column)
    ds = tf.data.Dataset.from_tensor_slices(series).repeat(4)  # creates a dataset with a separate element for each row of the input tensor
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(16).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# reading the dataset
df = pd.read_csv("csiro_alt_gmsl_mo_2015_csv_edited.csv")
df['id'] = df.index
time = df.index
series = df.GMSL

# setting up dataset
split = 180
xTrain = time[:split]
yTrain = series[:split]
xTest = time[split:]
yTest = series[split:]

window_size = 5
train_set = seq2seq_window_dataset(yTrain, window_size)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1])) # unknown rows, 1 column

for dilation_rate in (1, 2, 4, 8):
    model.add(keras.layers.Conv1D(filters=32, # number of output filters used in the convolution
                                  kernel_size=2, # Specifies the size of the convolutional window
                                  strides=1, # specifying the shift size
                                  dilation_rate=dilation_rate,
                                  padding="causal", # convolution output can be the same size as the input
                                  activation="relu"))
# model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(25, activation="relu"))
model.add(keras.layers.Dense(1))


optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

trained = model.fit(train_set, epochs=50)

cnn_forecast = model_forecast(model, series[:, np.newaxis], window_size)
#print(cnn_forecast)
cnn_forecast = cnn_forecast[split - window_size:-1, -1, 0]
print(cnn_forecast)

# list all data in history
# print(trained.history.keys())

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

axs[1].plot(xTrain, yTrain, label='Training Data')
axs[1].plot(xTest, yTest, label='Test Data')
axs[1].plot(xTest, cnn_forecast, label='Predicted Data')
axs[1].legend(loc='upper left', frameon=True)
axs[1].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Forecasting")
plt.show()

# error calculation
errors = cnn_forecast - yTest
abs_errors = np.abs(errors)
mae = abs_errors.mean()
print("Mean Absolute Error", mae)

# forecasting plot
fig, axs = plt.subplots(1)
plt.grid(True)
axs.plot(xTrain, yTrain, label='Training Data')
axs.plot(xTest, yTest, label='Test Data')
axs.plot(xTest, cnn_forecast, label='Predicted Data')
axs.legend(loc='upper left', frameon=True)
axs.set(ylabel='Sea Level Rise (mm)', xlabel='1993-December to 2015-January', title="Forecasting")

plt.show()
