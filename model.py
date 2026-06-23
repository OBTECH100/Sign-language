"""
model.py
--------
Defines the custom CNN architecture used for ASL alphabet classification.

Architecture rationale (useful for your methodology chapter):
- 4 convolutional blocks of increasing depth (32 -> 64 -> 128 -> 256
  filters) let the network learn low-level edge/texture features early
  and progressively more abstract hand-shape features in deeper layers.
- BatchNormalization after each conv layer stabilizes and speeds up
  training.
- Dropout at increasing rates reduces overfitting, which matters a lot
  here since many ASL Alphabet images share near-identical backgrounds.
- GlobalAveragePooling2D (instead of Flatten) keeps the parameter count
  of the dense head small and makes the network more robust to the
  exact hand position within the frame.

Run directly to print a model summary:
    python model.py
"""

from tensorflow.keras import layers, models, regularizers

import config


def build_cnn(input_shape=None, num_classes=None):
    input_shape = input_shape or (config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)
    num_classes = num_classes or config.NUM_CLASSES

    model = models.Sequential(name="ASL_CNN")
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.30))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.30))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
