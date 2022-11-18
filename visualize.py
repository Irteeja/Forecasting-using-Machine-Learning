import keras;
from keras.models import Sequential;
from keras.layers import Dense;

model = Sequential();
model.add(keras.layers.Dense(10, activation="relu", input_shape=[5]))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dense(1, activation="relu"))

from ann_visualizer.visualize import ann_viz;
ann_viz(model, title="");