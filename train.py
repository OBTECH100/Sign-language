"""
train.py
--------
Trains the CNN on the ASL Alphabet dataset and saves:
  - models/asl_cnn_best.h5   (best checkpoint by val_accuracy)
  - models/asl_cnn_final.h5  (final weights after early stopping restore)
  - outputs/training_curves.png
  - outputs/history.json
  - outputs/logs/training_log.csv

Run:
    python train.py
"""

import os
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
import matplotlib.pyplot as plt

import config
from data_preprocessing import get_datasets
from model import build_cnn


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    train_ds, val_ds, class_names = get_datasets()
    print(f"Training on {len(class_names)} classes.")

    model = build_cnn(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1
        ),
        CSVLogger(os.path.join(config.LOG_DIR, "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    model.save(config.FINAL_MODEL_PATH)
    print(f"Final model saved to {config.FINAL_MODEL_PATH}")

    plot_history(history, os.path.join(config.OUTPUT_DIR, "training_curves.png"))

    with open(os.path.join(config.OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)


if __name__ == "__main__":
    main()
