# ASL Sign Language Recognition System (CNN)

Real-time American Sign Language (ASL) alphabet recognition: webcam -> hand
detection -> CNN classification -> text output.

## Project structure

```
asl_recognition_project/
├── config.py                 # all paths & hyperparameters
├── data_preprocessing.py     # dataset loading, augmentation, normalization
├── model.py                  # CNN architecture
├── train.py                  # training loop + callbacks + curves
├── evaluate.py                # classification report + confusion matrix
├── hand_detector.py          # MediaPipe-based hand detection/segmentation
├── real_time_inference.py    # full real-time webcam app
├── predict_image.py          # test the model on a single static image
├── requirements.txt
├── data/
│   └── asl_alphabet_train/   # <- put the dataset here (see below)
├── models/                   # trained .h5 models are saved here
└── outputs/                  # plots, reports, logs, class_names.json
```

## 1. Setup (VS Code, Windows/Mac/Linux)

Open this folder in VS Code, then in the integrated terminal:

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

In VS Code, select this venv as the interpreter (Ctrl+Shift+P -> "Python:
Select Interpreter").

> Note: `mediapipe` requires Python 3.9–3.11. If you're on a newer Python
> version and installation fails, create the venv with `python3.11 -m venv venv`
> instead, or drop the version pins in requirements.txt and let pip resolve
> compatible versions.

## 2. Get the dataset

Download the **ASL Alphabet** dataset from Kaggle (search "ASL Alphabet" by
grassknoted), which contains A–Z, `space`, `del`, and `nothing` classes.

Unzip it so the structure looks like:

```
data/
└── asl_alphabet_train/
    ├── A/
    ├── B/
    ├── ...
    ├── Z/
    ├── space/
    ├── del/
    └── nothing/
```

Each folder should contain only images of that class. If the dataset ships
with an extra nested folder (e.g. `asl_alphabet_train/asl_alphabet_train/A/`),
move the class folders up one level so they sit directly under
`data/asl_alphabet_train/`.

## 3. Train the model

```bash
python train.py
```

This will:
- Load and split the dataset (85% train / 15% validation)
- Save `outputs/class_names.json` (the label mapping used everywhere else)
- Train the CNN with early stopping, LR reduction, and checkpointing
- Save the best model to `models/asl_cnn_best.h5`
- Save training curves to `outputs/training_curves.png`

Training the full dataset (29 classes x ~3000 images) on CPU will be slow
(hours). If you don't have a GPU, consider training on a subset first to
verify the pipeline works end-to-end, or use Google Colab for the training
step only, then download the resulting `.h5` file into `models/` for the
real-time app to use locally.

## 4. Evaluate the model

```bash
python evaluate.py
```

Produces `outputs/classification_report.txt`, `.json`, and
`outputs/confusion_matrix.png` — exactly what you need for the "Results and
Performance Analysis" chapter (accuracy, precision, recall, F1 per class).

## 5. Run real-time recognition

```bash
python real_time_inference.py
```

- Show your hand to the webcam; MediaPipe detects and boxes it.
- The cropped hand region is classified by the CNN every frame.
- Predictions are smoothed over several frames (majority vote) and a
  letter is only "typed" once it's held steadily — this avoids a jittery,
  flickering text output.
- Press `c` to clear the typed sentence, `q` to quit.

## 6. Quick single-image test

```bash
python predict_image.py path/to/image.jpg
```

Useful for grabbing clean demonstration screenshots for your report without
relying on the live webcam.

## Design notes (for your report)

- **Why no horizontal flip augmentation?** Flipping a hand gesture left-right
  can change its meaning or visually resemble a different letter, so it
  isn't a label-preserving augmentation here — unlike typical photo
  classification tasks.
- **Why GlobalAveragePooling2D instead of Flatten?** Keeps the dense head
  small and makes the classifier less sensitive to exactly where the hand
  sits within the cropped ROI.
- **Why MediaPipe over classical skin-color segmentation?** Skin-color/HSV
  thresholding is sensitive to lighting and skin tone and breaks down with
  cluttered/skin-colored backgrounds. MediaPipe Hands uses a pretrained
  landmark-detection model, giving a far more robust bounding box for
  real-time use — but you can mention classical HSV/contour segmentation in
  your literature review as the traditional alternative this design improves
  on.
- **Why majority-vote smoothing + a "hold" requirement before committing a
  letter?** Per-frame CNN predictions on a live, slightly shaky webcam feed
  flicker between classes. Smoothing and requiring a steady hold turns noisy
  frame-by-frame guesses into a usable, stable text stream.

## Next steps

Once training/evaluation results look good, we can move on to:
- Writing the Methodology and Implementation chapters around this exact code
- Writing the Results & Discussion chapter using your confusion matrix /
  classification report numbers
- Optionally comparing this custom CNN against a transfer-learning baseline
  (e.g. MobileNetV2) for a stronger "performance analysis" section
