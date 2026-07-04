"""
Synthetic validation of fretboard_calibration.py: since we don't have a real
guitar photo calibrated yet, this generates a fake fretboard image seen from
an angled (skewed) camera perspective, then verifies that calibrate() +
pixel_to_fret_string() correctly recovers known fret/string positions.
"""
import sys
sys.path.insert(0, '/home/claude/guitar-note-verifier')
import numpy as np
import cv2
from src.vision.fretboard_calibration import FretboardCalibration, fret_position_mm

SCALE_LENGTH = 647.7  # mm, standard 25.5" scale
N_FRETS = 24
N_STRINGS = 6
IMG_W, IMG_H = 1000, 300

# --- Step 1: define the "true" flat-space fretboard (bird's eye view) ---
# flat space: x = mm from nut, y = string index (0=low E ... 5=high E)
nut_to_12th = fret_position_mm(12, SCALE_LENGTH)
flat_corners = np.array([
    [0, 0],                    # nut, string 0 (low E)
    [nut_to_12th, 0],          # 12th fret, string 0
    [nut_to_12th, N_STRINGS - 1],  # 12th fret, string 5 (high E)
    [0, N_STRINGS - 1],        # nut, string 5
], dtype=np.float32)

# --- Step 2: simulate a camera viewing this region at an angle ---
# (a real photo would have perspective distortion -- we fake that here by
# picking a skewed quadrilateral in "pixel space" for these same 4 corners)
image_corners = np.array([
    [80, 60],     # nut/low-E appears up and to the right (angled view)
    [920, 20],    # 12th fret/low-E
    [960, 260],   # 12th fret/high-E
    [40, 280],    # nut/high-E
], dtype=np.float32)

# --- Step 3: calibrate ---
calib = FretboardCalibration(scale_length_mm=SCALE_LENGTH, n_frets=N_FRETS, n_strings=N_STRINGS)
calib.calibrate(image_corners, flat_corners)

# --- Step 4: pick several known (fret, string) test points, project them
# into the fake image space using the INVERSE of what we just calibrated,
# then confirm pixel_to_fret_string() recovers the original fret/string ---
test_cases = [
    (0, 0),   # open low E
    (3, 0),   # 3rd fret low E (G)
    (5, 2),   # 5th fret D string
    (12, 5),  # 12th fret high E
    (7, 3),   # 7th fret G string
    (1, 1),   # 1st fret A string
]

# homography mapping flat -> image (inverse direction), to synthesize fake
# pixel coordinates for each test case
flat_to_image_H, _ = cv2.findHomography(flat_corners, image_corners)

print(f"{'True Fret':>10} | {'True String':>11} | {'Pred Fret':>10} | {'Pred String':>11} | Match")
print("-" * 62)
all_correct = True
for (true_fret, true_string) in test_cases:
    flat_x = fret_position_mm(true_fret, SCALE_LENGTH)
    flat_pt = np.array([[[flat_x, true_string]]], dtype=np.float32)
    pixel_pt = cv2.perspectiveTransform(flat_pt, flat_to_image_H)[0, 0]

    pred_string, pred_fret = calib.pixel_to_fret_string(tuple(pixel_pt))
    match = (pred_fret == true_fret) and (pred_string == true_string)
    all_correct &= match
    print(f"{true_fret:>10} | {true_string:>11} | {pred_fret:>10} | {pred_string:>11} | {'OK' if match else 'MISMATCH'}")

print(f"\nAll test cases passed: {all_correct}")
