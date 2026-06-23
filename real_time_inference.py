"""
real_time_inference.py
------------------------
The full real-time pipeline: webcam capture -> hand detection/segmentation
(MediaPipe) -> ROI preprocessing -> CNN classification -> temporal
smoothing -> text output on screen.

Controls:
    q  - quit
    c  - clear the typed sentence

Run:
    python real_time_inference.py
"""

import os
import json
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf

import config
from hand_detector import HandDetector


def load_class_names():
    mapping_path = os.path.join(config.OUTPUT_DIR, "class_names.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"{mapping_path} not found. Run data_preprocessing.py or "
            "train.py at least once first -- they generate this file "
            "from your dataset's folder names."
        )
    with open(mapping_path, "r") as f:
        return json.load(f)


def preprocess_roi(roi_bgr, img_size):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (img_size, img_size))
    roi_norm = roi_resized.astype("float32") / 255.0
    return np.expand_dims(roi_norm, axis=0)


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = load_class_names()

    detector = HandDetector(padding=config.ROI_PADDING)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: could not open webcam. Check that it isn't in use "
              "by another application and that OpenCV has camera permission.")
        return

    sentence = ""
    pred_history = deque(maxlen=config.PREDICTION_SMOOTHING)
    stable_label = None
    stable_count = 0

    print("Webcam started. Press 'q' to quit, 'c' to clear the sentence.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural "looking in a mirror" feel
        box, landmarks = detector.find_hand_roi(frame)

        current_label, current_conf = None, 0.0

        if box is not None:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                input_tensor = preprocess_roi(roi, config.IMG_SIZE)
                preds = model.predict(input_tensor, verbose=0)[0]
                idx = int(np.argmax(preds))
                current_conf = float(preds[idx])
                current_label = class_names[idx]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detector.draw_landmarks(frame, landmarks)

        # ---- Temporal smoothing: majority vote over recent frames ----
        if current_label is not None and current_conf >= config.CONFIDENCE_THRESHOLD:
            pred_history.append(current_label)
        else:
            pred_history.append(None)

        if len(pred_history) == pred_history.maxlen:
            valid = [p for p in pred_history if p is not None]
            if valid:
                majority_label, count = Counter(valid).most_common(1)[0]
                if count >= int(0.6 * pred_history.maxlen):
                    if majority_label == stable_label:
                        stable_count += 1
                    else:
                        stable_label = majority_label
                        stable_count = 1
                else:
                    stable_label, stable_count = None, 0
            else:
                stable_label, stable_count = None, 0

        # ---- Commit a stable, held gesture into the output text ----
        if stable_label is not None and stable_count == config.HOLD_FRAMES_FOR_COMMIT:
            if stable_label == "space":
                sentence += " "
            elif stable_label == "del":
                sentence = sentence[:-1]
            elif stable_label != "nothing":
                sentence += stable_label
            stable_count = 0  # require a fresh hold before the same letter repeats

        # ---- On-screen UI ----
        display_pred = current_label if current_label else "No hand detected"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (30, 30, 30), -1)
        cv2.putText(frame, f"Prediction: {display_pred} ({current_conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Text: {sentence}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ASL Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
