import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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

xTrain = np.array(xTrain).reshape((-1, 1)) # rows unknown , column 1
yTrain = np.array(yTrain).reshape((-1, 1))
xTest = np.array(xTest).reshape((-1, 1))
yTest = np.array(yTest).reshape((-1, 1))

model = LinearRegression()
model.fit(xTrain, yTrain)
print((1-model.score(xTrain, yTrain))*100)

yPred = model.predict(xTest)

#print(yPred)

fig, axs = plt.subplots(2)
plt.grid(True)

axs[0].bar(time, series, alpha=1)
axs[0].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Original Data")

axs[1].plot(xTrain, yTrain, label='Training Data')
axs[1].plot(xTest, yTest, label='Test Data')
axs[1].plot(xTest, yPred, label='Predicted Data')
axs[1].legend(loc='upper left', frameon=True)
axs[1].set(ylabel='Sea Level Rise (mm)', xlabel='1993-2014', title="Forecasting")

plt.show()
