"""
config.py
---------
Single source of truth for paths and hyperparameters used across the
ASL Sign Language Recognition project. Edit values here rather than
hunting through individual scripts.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point this at the folder that contains one sub-folder per class
# (A, B, C, ..., Z, space, del, nothing), e.g. the Kaggle "ASL Alphabet"
# dataset's asl_alphabet_train directory.
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet_train")

# Optional separate test set (Kaggle's asl_alphabet_test has a handful
# of unlabeled images per class encoded in the filename). Leave as-is
# if you don't have one -- evaluate.py uses the validation split instead.
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet_test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Image / dataset parameters
# ----------------------------------------------------------------------
IMG_SIZE = 64            # all images resized to IMG_SIZE x IMG_SIZE
CHANNELS = 3
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.15
SEED = 42

# Standard Kaggle ASL Alphabet dataset: A-Z + space + del + nothing = 29 classes.
# This is overwritten dynamically from the dataset folder names at load time
# (see data_preprocessing.py), this list is just a fallback/reference.
CLASS_NAMES = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["space", "del", "nothing"]
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------------------------------------------------
# Training parameters
# ----------------------------------------------------------------------
EPOCHS = 40
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3

MODEL_PATH = os.path.join(MODEL_DIR, "asl_cnn_best.h5")          # best checkpoint
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "asl_cnn_final.h5")   # last epoch / restored best

# ----------------------------------------------------------------------
# Real-time inference parameters
# ----------------------------------------------------------------------
ROI_PADDING = 20              # pixels of padding added around the detected hand box
PREDICTION_SMOOTHING = 8      # number of recent frames used for majority-vote smoothing
CONFIDENCE_THRESHOLD = 0.70   # minimum softmax confidence to accept a prediction
HOLD_FRAMES_FOR_COMMIT = 15   # consecutive stable frames required before a letter is "typed"
