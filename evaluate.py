"""
evaluate.py
-----------
Loads the trained model and the validation split, then generates:
  - outputs/classification_report.txt / .json  (precision/recall/F1 per class)
  - outputs/confusion_matrix.png

Run:
    python evaluate.py
"""

import os
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import config
from data_preprocessing import get_datasets


def evaluate(model_path=None):
    model_path = model_path or config.MODEL_PATH
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    _, val_ds, class_names = get_datasets()

    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_accuracy = float(np.mean(y_true == y_pred))
    print(f"Overall validation accuracy: {overall_accuracy:.4f}")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print(report_text)

    with open(os.path.join(config.OUTPUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(os.path.join(config.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Overall accuracy: {overall_accuracy:.4f}\n\n")
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - ASL CNN Classifier")
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    evaluate()
