# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
# -*- coding: utf-8 -*-
"""
CSE 535 - SmartHome Gesture Control Application (Project Part 2)

This script:
1) Extracts the middle frame of each training/test video
2) Uses the provided CNN (cnn_model.h5) via HandShapeFeatureExtractor to get feature vectors
3) Classifies each test video by nearest-neighbor cosine distance against training vectors
4) Writes predictions to Results.csv (NO header, 51x1, integers only)

Folder expectations (relative to this file):
- traindata/   (your training gesture videos from Part 1)
- test/        (autograder will place test videos here)
- handshape_feature_extractor.py, cnn_model.h5 (provided)

IMPORTANT: Do not use absolute paths.
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from handshape_feature_extractor import HandShapeFeatureExtractor


# -----------------------------
# Label parsing (UPDATED for your exact filenames)
# -----------------------------

SMART_HOME_NAME_TO_LABEL = {
    "DecreaseFanSpeed": 10,
    "DecereaseFanSpeed": 10,  # tolerate common typo
    "FanOff": 11,
    "FanOn": 12,
    "IncreaseFanSpeed": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16,
}

def infer_label_from_path(path: str) -> int:
    """
    Your training files look like:
      H-0.mp4 ... H-9.mp4
      H-FanOn.mp4, H-LightOff.mp4, H-DecreaseFanSpeed.mp4, etc.

    This function extracts the label deterministically from that pattern.
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)

    # Strip "H-" prefix if present
    if stem.startswith("H-"):
        stem = stem[2:]

    # Numeric gestures 0..9
    if stem.isdigit():
        lab = int(stem)
        if 0 <= lab <= 9:
            return lab

    # Smart-home gestures 10..16
    if stem in SMART_HOME_NAME_TO_LABEL:
        return SMART_HOME_NAME_TO_LABEL[stem]

    raise ValueError(
        f"Could not infer training label from filename: {base}\n"
        f"Expected patterns like 'H-3.mp4' or 'H-LightOn.mp4'."
    )


# -----------------------------
# Video / feature helpers
# -----------------------------

def list_videos(root_dir: str) -> List[str]:
    """Recursively list common video files under root_dir, sorted by relative path."""
    exts = (".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v")
    paths: List[str] = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(r, f))
    paths.sort(key=lambda p: os.path.relpath(p, root_dir))
    return paths

def extract_middle_frame_gray(video_path: str) -> np.ndarray:
    """
    Extract the middle frame from a video and return it as a grayscale (H,W) uint8 array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame = None
    if frame_count and frame_count > 0:
        mid = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if (not ok) or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, mid - 1))
            ok, frame = cap.read()
    else:
        # Fallback: read all frames (slower but robust)
        frames = []
        ok, f = cap.read()
        while ok and f is not None:
            frames.append(f)
            ok, f = cap.read()
        if frames:
            frame = frames[len(frames) // 2]

    cap.release()

    if frame is None:
        raise RuntimeError(f"Failed to read middle frame from: {video_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def extract_feature_vector(feature_extractor: HandShapeFeatureExtractor, gray_frame: np.ndarray) -> np.ndarray:
    """
    Run the CNN feature extractor and return a 1D float vector.
    """
    vec = feature_extractor.extract_feature(gray_frame)
    vec = np.array(vec).reshape(-1).astype(np.float32)
    return vec


# -----------------------------
# Main pipeline
# -----------------------------

def build_training_bank(train_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - train_vectors: (N, D)
    - train_labels:  (N,)
    """
    train_videos = list_videos(train_dir)
    if not train_videos:
        raise FileNotFoundError(
            f"No training videos found under '{train_dir}'. "
            f"Place your Part 1 training gesture videos inside traindata/."
        )

    extractor = HandShapeFeatureExtractor.get_instance()

    vectors: List[np.ndarray] = []
    labels: List[int] = []

    for vp in train_videos:
        lab = infer_label_from_path(vp)
        gray = extract_middle_frame_gray(vp)
        vec = extract_feature_vector(extractor, gray)
        vectors.append(vec)
        labels.append(lab)

    train_vectors = np.vstack(vectors)
    train_labels = np.array(labels, dtype=np.int32)
    return train_vectors, train_labels

def predict_test_labels(train_vectors: np.ndarray, train_labels: np.ndarray, test_dir: str) -> np.ndarray:
    """
    Predict labels for each test video in sorted order and return a (M,) int array.
    """
    test_videos = list_videos(test_dir)
    if not test_videos:
        raise FileNotFoundError(
            f"No test videos found under '{test_dir}'. "
            f"During autograding, the 'test/' folder is provided automatically."
        )

    extractor = HandShapeFeatureExtractor.get_instance()

    # Precompute training norms once for speed
    A = train_vectors
    A_norms = np.linalg.norm(A, axis=1) + 1e-8

    preds: List[int] = []

    for vp in test_videos:
        gray = extract_middle_frame_gray(vp)
        test_vec = extract_feature_vector(extractor, gray)

        b_norm = np.linalg.norm(test_vec) + 1e-8
        sims = (A @ test_vec) / (A_norms * b_norm)   # cosine similarity
        dists = 1.0 - sims                            # cosine distance
        best_idx = int(np.argmin(dists))
        preds.append(int(train_labels[best_idx]))

    return np.array(preds, dtype=np.int32)

def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(root, "traindata")
    test_dir = os.path.join(root, "test")
    out_path = os.path.join(root, "Results.csv")

    train_vectors, train_labels = build_training_bank(train_dir)
    preds = predict_test_labels(train_vectors, train_labels, test_dir)

    # Autograder expects NO header and integers only (51 x 1).
    np.savetxt(out_path, preds.reshape(-1, 1), fmt="%d", delimiter=",")

    print(f"Wrote {preds.shape[0]} predictions to {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
