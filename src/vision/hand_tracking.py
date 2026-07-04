"""
src/vision/hand_tracking.py

Detects hand landmarks (21 points per hand: fingertips, knuckles, wrist) in
a photo using MediaPipe's HandLandmarker. Run this locally (not in a
network-restricted sandbox) since it needs to download a small model file
from Google on first use.

Usage:
    python3 src/vision/hand_tracking.py path/to/hand_photo.jpg

Output: prints detected landmark pixel coordinates and saves an annotated
image (<input>_landmarks.png) showing the detected hand skeleton.
"""
import sys
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Fingertip landmark indices in MediaPipe's 21-point hand model
FINGERTIP_IDS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def detect_hand_landmarks(image_path):
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img_bgr.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    hands_pixels = []
    for hand_landmarks in result.hand_landmarks:
        pixels = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
        hands_pixels.append(pixels)

    return hands_pixels, img_bgr


def draw_landmarks(img_bgr, hands_pixels):
    vis = img_bgr.copy()
    for pixels in hands_pixels:
        for (a, b) in HAND_CONNECTIONS:
            pa = tuple(map(int, pixels[a]))
            pb = tuple(map(int, pixels[b]))
            cv2.line(vis, pa, pb, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pixels):
            color = (0, 0, 255) if i in FINGERTIP_IDS.values() else (255, 255, 0)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 hand_tracking.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    hands_pixels, img_bgr = detect_hand_landmarks(image_path)

    print(f"\nDetected {len(hands_pixels)} hand(s)")
    for i, pixels in enumerate(hands_pixels):
        print(f"\nHand {i+1} fingertip positions (pixel x, y):")
        for name, idx in FINGERTIP_IDS.items():
            x, y = pixels[idx]
            print(f"  {name:>12}: ({x:.0f}, {y:.0f})")

    vis = draw_landmarks(img_bgr, hands_pixels)
    out_path = os.path.splitext(image_path)[0] + "_landmarks.png"
    cv2.imwrite(out_path, vis)
    print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
