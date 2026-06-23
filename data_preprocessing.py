"""
data_preprocessing.py
----------------------
Loads the ASL Alphabet dataset from disk, splits it into train/validation
sets, applies normalization and (train-only) data augmentation, and
returns ready-to-use tf.data.Dataset objects.

Run directly to sanity-check that the dataset loads correctly:
    python data_preprocessing.py
"""

import os
import json

import tensorflow as tf
from tensorflow.keras import layers

import config


def get_datasets():
    """
    Returns (train_ds, val_ds, class_names).

    train_ds / val_ds: batched, normalized, prefetched tf.data.Dataset
                        objects yielding (images, one_hot_labels).
    class_names: list of class names in the index order used by the model
                 (also saved to outputs/class_names.json for the inference
                 app to load later).
    """
    if not os.path.isdir(config.DATA_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found at: {config.DATA_DIR}\n"
            "Download the ASL Alphabet dataset and place its class "
            "sub-folders (A, B, C, ..., space, del, nothing) inside "
            "this path. See README.md for instructions."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    # Persist the class index mapping so real_time_inference.py and
    # predict_image.py can decode model outputs consistently later.
    mapping_path = os.path.join(config.OUTPUT_DIR, "class_names.json")
    with open(mapping_path, "w") as f:
        json.dump(class_names, f, indent=2)

    normalization = layers.Rescaling(1.0 / 255)

    # NOTE: deliberately no horizontal flip here -- flipping a hand
    # gesture left-right can turn one ASL letter into a visually
    # different (or confusing) one, so it is not a label-preserving
    # augmentation for this task.
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])

    def prep_train(x, y):
        x = augmentation(x, training=True)
        x = normalization(x)
        return x, y

    def prep_val(x, y):
        x = normalization(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    train_ds, val_ds, class_names = get_datasets()
    print(f"Detected {len(class_names)} classes:")
    print(class_names)
    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
