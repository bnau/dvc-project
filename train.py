from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

train_features = np.load("preprocessed/train_features.npy")
train_labels = np.load("preprocessed/train_labels.npy")
val_features = np.load("preprocessed/val_features.npy")
val_labels = np.load("preprocessed/val_labels.npy")

inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

model.fit(train_features, train_labels,
          epochs=20,
          validation_data=(val_features, val_labels))

model.save("model.h5")
