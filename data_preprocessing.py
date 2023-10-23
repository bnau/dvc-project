from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from pathlib import Path

new_base_dir = Path("subset")
train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

base_model = keras.models.load_model("base.h5")

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = preprocess_input(
            images, mode="caffe"
        )
        features = base_model.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

Path("preprocessed").mkdir(exist_ok=True)
np.save("preprocessed/train_features.npy", train_features)
np.save("preprocessed/train_labels.npy", train_labels)
np.save("preprocessed/val_features.npy", val_features)
np.save("preprocessed/val_labels.npy", val_labels)
np.save("preprocessed/test_features.npy", test_features)
np.save("preprocessed/test_labels.npy", test_labels)
