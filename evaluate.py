from tensorflow import keras
import numpy as np

test_features = np.load("preprocessed/test_features.npy")
test_labels = np.load("preprocessed/test_labels.npy")

test_model = keras.models.load_model("model.h5")
test_loss, test_acc = test_model.evaluate(test_features, test_labels)

with open("metrics.json", "w", encoding="utf-8") as f:
    f.write(f"{{\"test_accuracy\": {test_acc}}}\n")
