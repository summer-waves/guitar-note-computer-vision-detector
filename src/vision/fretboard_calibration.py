"""
src/vision/fretboard_calibration.py

Maps pixel coordinates from a camera view of a guitar neck into
(string_index, fret_number) using a one-time perspective calibration.

Key physics detail this accounts for: frets are NOT evenly spaced. Real
fret position follows the equal-tempered spacing rule -- each fret is
positioned at a fraction of the remaining string length. We use this to
convert a "flat" (bird's-eye) distance-from-nut into the correct fret
number, rather than assuming a naive linear grid.

Calibration workflow (one-time per camera setup):
  1. Click/identify 4 pixel points in the image: nut top, nut bottom,
     bridge-side reference top, bridge-side reference bottom (or any two
     fret positions along the neck -- see calibrate()).
  2. calibrate() computes a homography mapping those pixels to a flat
     real-world coordinate system (mm along neck, string index).
  3. pixel_to_fret_string() uses that homography for any fingertip pixel.
"""
import numpy as np
import cv2

N_STRINGS = 6


def fret_position_mm(fret_number, scale_length_mm):
    """
    Distance from the nut to a given fret, in mm, using the standard
    equal-tempered fret spacing formula. fret_number=0 is the nut itself.
    """
    if fret_number == 0:
        return 0.0
    return scale_length_mm * (1 - 1 / (2 ** (fret_number / 12.0)))


def build_fret_table(scale_length_mm, n_frets=24):
    """Precompute (fret_number -> distance_from_nut_mm) for fast lookup."""
    return {f: fret_position_mm(f, scale_length_mm) for f in range(n_frets + 1)}


def nearest_fret(distance_from_nut_mm, fret_table):
    """Given a flat-space distance from the nut, find the closest fret number."""
    frets = list(fret_table.keys())
    dists = [abs(fret_table[f] - distance_from_nut_mm) for f in frets]
    return frets[int(np.argmin(dists))]


class FretboardCalibration:
    """
    Holds a calibrated homography for one camera setup + one guitar's
    scale length, and converts pixel coordinates to (string, fret).
    """

    def __init__(self, scale_length_mm=647.7, n_frets=24, n_strings=N_STRINGS):
        # 647.7mm ~= 25.5" scale length, the Fender/Stratocaster standard.
        # Gibson-style guitars are closer to 628mm (24.75"); classical
        # guitars ~650mm. Pass the correct value for your instrument.
        self.scale_length_mm = scale_length_mm
        self.n_frets = n_frets
        self.n_strings = n_strings
        self.fret_table = build_fret_table(scale_length_mm, n_frets)
        self.homography = None

    def calibrate(self, image_points, flat_points):
        """
        image_points: list of 4 (x, y) pixel coordinates in the camera image,
            corresponding to 4 known reference points on the fretboard
            (e.g. the four corners of the region from the nut to the 12th fret).
        flat_points: the corresponding 4 (distance_from_nut_mm, string_pos)
            coordinates in "flat" real-world space for those same 4 points.
            string_pos runs 0 (e.g. low E) to n_strings-1 (high E), and the
            corner points should use 0 and n_strings-1 for the top/bottom edge.

        Computes and stores the homography mapping image pixels -> flat space.
        """
        img_pts = np.array(image_points, dtype=np.float32)
        flat_pts = np.array(flat_points, dtype=np.float32)
        self.homography, _ = cv2.findHomography(img_pts, flat_pts, method=0)
        return self.homography

    def pixel_to_flat(self, pixel_xy):
        """Map a single (x, y) pixel to (distance_from_nut_mm, string_pos_float)."""
        if self.homography is None:
            raise RuntimeError("Call calibrate() first")
        pt = np.array([[pixel_xy]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.homography)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def pixel_to_fret_string(self, pixel_xy):
        """
        Map a pixel coordinate to the nearest (string_index, fret_number).
        string_index is rounded to the nearest integer string (0-based);
        fret_number accounts for non-linear fret spacing via the fret table.
        """
        dist_mm, string_pos = self.pixel_to_flat(pixel_xy)
        fret = nearest_fret(dist_mm, self.fret_table)
        string_index = int(round(np.clip(string_pos, 0, self.n_strings - 1)))
        return string_index, fret
