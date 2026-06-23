"""
predict_image.py
------------------
Quick utility to test the trained model on a single static image file
(useful for taking demonstration screenshots for your report, or for
sanity-checking the model outside the webcam loop).

Run:
    python predict_image.py path/to/some_hand_image.jpg
"""

import sys
import os
import json

import numpy as np
import cv2
import tensorflow as tf

import config


def load_class_names():
    mapping_path = os.path.join(config.OUTPUT_DIR, "class_names.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def predict(image_path, model_path=None):
    model_path = model_path or config.MODEL_PATH
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0)

    preds = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    print(f"Predicted: {class_names[idx]}  (confidence: {preds[idx]:.4f})")

    top3_idx = np.argsort(preds)[-3:][::-1]
    print("Top-3 predictions:")
    for i in top3_idx:
        print(f"  {class_names[i]}: {preds[i]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])
