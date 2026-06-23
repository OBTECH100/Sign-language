"""
hand_detector.py
-----------------
Wraps MediaPipe Hands to detect a hand in a video frame and return a
padded bounding box (ROI) for cropping, plus the landmarks for on-screen
visualization. This is the "hand detection and segmentation" stage of
the pipeline: it isolates the hand region from the background before
the cropped ROI is preprocessed and passed to the CNN.
"""

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, padding=20, detection_confidence=0.7, tracking_confidence=0.6):
        self.padding = padding
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hand_roi(self, frame):
        """
        Returns (roi_box, landmarks).
        roi_box = (x1, y1, x2, y2) pixel coordinates of the padded hand
        bounding box, or (None, None) if no hand is detected in the frame.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]

        x1 = max(int(min(xs)) - self.padding, 0)
        y1 = max(int(min(ys)) - self.padding, 0)
        x2 = min(int(max(xs)) + self.padding, w)
        y2 = min(int(max(ys)) + self.padding, h)

        return (x1, y1, x2, y2), landmarks

    def draw_landmarks(self, frame, landmarks):
        self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self):
        self.hands.close()
