"""
src/vision/pose_tracking.py

Detects body pose landmarks (33 points: shoulders, hips, ears, nose, etc.)
using MediaPipe's PoseLandmarker. Mirrors hand_tracking.py's structure.
Run locally (not in a network-restricted sandbox) since it needs to
download a model file from Google on first use.

This module ONLY detects landmark positions. Posture "correct/incorrect"
judgment logic is intentionally kept separate (see posture_analysis.py)
since that involves heuristic thresholds, not just geometry.

Usage:
    python3 src/vision/pose_tracking.py path/to/photo.jpg

Output: prints key landmark pixel coordinates and saves an annotated image.
"""
import sys
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

# Key landmark indices in MediaPipe's 33-point pose model (BlazePose topology).
# Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_LANDMARKS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

# A minimal skeleton for visualization -- just the upper body, since that's
# what matters for guitar posture (we're not tracking legs/feet here)
POSE_CONNECTIONS = [
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmark model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def detect_pose_landmarks(image_path):
    """
    Returns (landmarks_dict_list, img_bgr).
    landmarks_dict_list: list of dicts (one per detected person), each
    mapping landmark name -> (x_pixel, y_pixel) for the names in POSE_LANDMARKS.
    """
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img_bgr.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    all_poses = []
    for pose_landmarks in result.pose_landmarks:
        named = {}
        for name, idx in POSE_LANDMARKS.items():
            lm = pose_landmarks[idx]
            named[name] = (lm.x * w, lm.y * h)
        all_poses.append(named)

    return all_poses, img_bgr


def draw_pose(img_bgr, all_poses):
    vis = img_bgr.copy()
    for pose in all_poses:
        for (a, b) in POSE_CONNECTIONS:
            if a in pose and b in pose:
                pa = tuple(map(int, pose[a]))
                pb = tuple(map(int, pose[b]))
                cv2.line(vis, pa, pb, (0, 255, 0), 2)
        for name, (x, y) in pose.items():
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, name, (int(x) + 8, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pose_tracking.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    all_poses, img_bgr = detect_pose_landmarks(image_path)

    print(f"\nDetected {len(all_poses)} pose(s)")
    for i, pose in enumerate(all_poses):
        print(f"\nPose {i+1} key landmarks (pixel x, y):")
        for name, (x, y) in pose.items():
            print(f"  {name:>15}: ({x:.0f}, {y:.0f})")

    vis = draw_pose(img_bgr, all_poses)
    out_path = os.path.splitext(image_path)[0] + "_pose.png"
    cv2.imwrite(out_path, vis)
    print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
