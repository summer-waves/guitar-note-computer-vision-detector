"""
src/vision/detect_frets.py

Given a cropped photo of a guitar neck (nut visible at top), automatically
detects the fretboard position markers (the inlay dots), matches them to
known fret numbers using the standard marker pattern, and fits the
equal-tempered fret spacing model to find the pixel-space nut position and
scale. This calibration can then feed fretboard_calibration.py to map any
fingertip pixel to a (string, fret) position.

Usage:
    python3 src/vision/detect_frets.py path/to/cropped_neck.jpg

Note: this expects the input image to already be roughly cropped to just
the neck (nut near the top, frets running down/across). Cropping out the
body/headstock isn't automated yet -- that's the next thing to build once
we have a few more real photos to test against.
"""
import sys
import numpy as np
import cv2

# Standard single-dot marker frets, and the one double-dot fret, for a
# typical 20ish-fret acoustic guitar. Order matters: this is the sequence
# we expect to see top-to-bottom in a neck photo.
EXPECTED_MARKER_FRETS = [3, 5, 7, 9, 12, 15, 17, 19, 21]


def detect_marker_rows(gray_neck_image, side_margin_frac=0.2):
    """
    Detect circular position-marker dots in a grayscale cropped neck image.
    Returns a sorted list of row (y) pixel positions, one per detected dot
    (a fret with a double-dot inlay will produce two close rows -- caller
    should collapse those).
    """
    h, w = gray_neck_image.shape
    margin = int(w * side_margin_frac)
    central = gray_neck_image[:, margin:w - margin]
    blurred = cv2.GaussianBlur(central, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
        param1=60, param2=18, minRadius=4, maxRadius=14
    )

    if circles is None:
        return []

    rows = sorted(float(y) for (x, y, r) in np.round(circles[0, :]).astype(int))
    return rows


def collapse_double_dots(rows, merge_thresh_px=15):
    """Merge rows that are within merge_thresh_px of each other (double-dot fret)."""
    if not rows:
        return []
    merged = [[rows[0]]]
    for r in rows[1:]:
        if r - merged[-1][-1] <= merge_thresh_px:
            merged[-1].append(r)
        else:
            merged.append([r])
    return [np.mean(group) for group in merged]


def match_rows_to_frets(collapsed_rows, expected_frets=EXPECTED_MARKER_FRETS):
    """
    Assumes collapsed_rows are sorted top-to-bottom and correspond, in order,
    to the first N entries of expected_frets. This is a simple positional
    match -- good enough when the photo cleanly shows the marker sequence
    starting from fret 3, but won't handle a photo that starts mid-neck.
    """
    n = min(len(collapsed_rows), len(expected_frets))
    return {expected_frets[i]: collapsed_rows[i] for i in range(n)}


def fit_nut_and_scale(fret_to_row):
    """
    Fit row = row_nut + scale_px * (1 - 2^(-fret/12)) via least squares.
    Returns (row_nut, scale_px, r_squared, per_fret_errors_px).
    """
    frets = np.array(list(fret_to_row.keys()))
    rows = np.array(list(fret_to_row.values()), dtype=float)
    rel_dist = 1 - 2 ** (-frets / 12.0)

    A = np.vstack([np.ones_like(rel_dist), rel_dist]).T
    (row_nut, scale_px), _, _, _ = np.linalg.lstsq(A, rows, rcond=None)

    predicted = row_nut + scale_px * rel_dist
    errors = predicted - rows
    r_squared = 1 - np.sum(errors ** 2) / np.sum((rows - rows.mean()) ** 2)

    return float(row_nut), float(scale_px), float(r_squared), dict(zip(frets.tolist(), errors.tolist()))


def calibrate_from_neck_photo(image_path):
    """
    End-to-end: load a cropped neck photo, detect markers, fit the model.
    Returns a dict with row_nut, scale_px, r_squared, and the fret->row map
    used for fitting -- or None if not enough markers were detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw_rows = detect_marker_rows(gray)
    collapsed = collapse_double_dots(raw_rows)
    fret_to_row = match_rows_to_frets(collapsed)

    if len(fret_to_row) < 3:
        return None

    row_nut, scale_px, r_squared, errors = fit_nut_and_scale(fret_to_row)

    return {
        "row_nut": row_nut,
        "scale_px": scale_px,
        "r_squared": r_squared,
        "fret_to_row": fret_to_row,
        "fit_errors_px": errors,
        "image_shape": img.shape,
    }


def fret_row(fret_num, row_nut, scale_px):
    """Given a fitted calibration, return the pixel row for any fret number."""
    rel_dist = 1 - 2 ** (-fret_num / 12.0) if fret_num > 0 else 0.0
    return row_nut + scale_px * rel_dist


def estimate_fret_from_row(row, row_nut, scale_px, max_fret=24):
    """
    Inverse of fret_row(): given a pixel row (e.g. a detected fingertip's y
    coordinate) and a fitted calibration, estimate which fret that row is
    closest to.

    Derivation: row = row_nut + scale_px * (1 - 2^(-n/12))
        => (row - row_nut) / scale_px = 1 - 2^(-n/12)
        => n = -12 * log2(1 - rel_dist)

    Returns the nearest integer fret number, clipped to [0, max_fret].
    Note: this only uses the vertical (along-neck) position -- it does not
    determine which string the finger is on, since that requires a full
    2D homography (see fretboard_calibration.py) rather than this 1D
    nut/scale fit. Combine with that module for full (string, fret) output.
    """
    rel_dist = (row - row_nut) / scale_px
    rel_dist = min(max(rel_dist, -0.5), 0.999)  # guard against log domain errors
    if rel_dist <= 0:
        return 0
    n = -12 * np.log2(1 - rel_dist)
    return int(np.clip(round(n), 0, max_fret))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_frets.py <cropped_neck_image>")
        sys.exit(1)

    result = calibrate_from_neck_photo(sys.argv[1])
    if result is None:
        print("Not enough position markers detected to calibrate. "
              "Make sure the photo clearly shows the neck with visible fret markers.")
        sys.exit(1)

    print(f"Fitted nut row (px):    {result['row_nut']:.1f}")
    print(f"Fitted scale (px):      {result['scale_px']:.1f}")
    print(f"Fit quality (R^2):      {result['r_squared']:.5f}")
    print(f"\nMarkers used: {result['fret_to_row']}")


if __name__ == "__main__":
    main()