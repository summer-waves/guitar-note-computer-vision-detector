"""
src/vision/posture_analysis.py

Computes posture-related angles from pose landmarks (see pose_tracking.py).
Deliberately kept separate from landmark detection: this module contains
heuristic/reference-based judgment calls, while pose_tracking.py contains
only geometric fact (landmark positions).

IMPORTANT SCOPE NOTE: this is general postural guidance adapted from
published ergonomics research, NOT a medical assessment or diagnosis of
any individual. It should not be treated as clinical advice. The
craniovertebral angle threshold used below is commonly cited in the
ergonomics/physical therapy literature, but researchers note there is no
universal consensus on the exact cutoff -- see the docstring on
craniovertebral_angle() for sourcing.
"""
import numpy as np


def _angle_from_horizontal(p_from, p_to):
    """
    Angle in degrees between the horizontal axis and the line from p_from
    to p_to, measured as a positive value regardless of direction.
    p_from, p_to: (x, y) pixel tuples. Note: image y-axis increases downward.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    angle_rad = np.arctan2(abs(dy), abs(dx))
    return np.degrees(angle_rad)


def craniovertebral_angle(shoulder, ear):
    """
    Craniovertebral angle (CVA): angle between a horizontal line through the
    shoulder and the line from the shoulder to the ear. Used in ergonomics/
    physical therapy research as a proxy for forward head posture -- a
    smaller angle indicates the head is positioned further forward relative
    to the shoulders.

    Commonly cited reference point: a standing CVA below approximately
    50 degrees is often described in the literature as indicating forward
    head posture, though there is no single universal cutoff agreed upon
    across studies (some cite 48, others 50-52 degrees). Treat this as
    general postural guidance, not a diagnostic threshold.

    IMPORTANT LIMITATION: the real CVA measurement protocol requires a
    photo taken from directly the side (sagittal/profile view), since it's
    measuring how far forward the head juts relative to the shoulder in
    the front-to-back direction. A front-facing photo (like the one this
    project has tested against so far) does NOT capture that -- it instead
    measures ear height relative to shoulder height from the front, which
    is a related but different thing. Don't treat a front-photo result as
    a valid CVA reading; a side-profile photo is needed for that.

    shoulder, ear: (x, y) pixel tuples from pose_tracking.py landmarks.
    Returns the angle in degrees.
    """
    return _angle_from_horizontal(shoulder, ear)


def shoulder_tilt_angle(left_shoulder, right_shoulder):
    """
    Angle of the shoulder line relative to horizontal. 0 degrees means
    perfectly level shoulders. Persistent asymmetry here can result from
    uneven guitar strap tension or habitually favoring one side.
    """
    return _angle_from_horizontal(left_shoulder, right_shoulder)


def guitar_neck_angle(nut_point, body_point):
    """
    Angle of the guitar neck relative to horizontal, given two pixel points
    along the neck (e.g. the nut and a point near where the neck meets the
    body -- these currently need to be identified manually or from a
    separate neck-detection step; there is no automatic full-photo guitar
    detection yet, see detect_frets.py's known limitations).

    This is purely descriptive (what angle is the guitar actually held at),
    not evaluative -- correct guitar angle varies significantly by playing
    style (classical vs. casual/rock posture) and is a matter of technique
    convention, not a single "correct" number.
    """
    return _angle_from_horizontal(nut_point, body_point)


def analyze_posture(pose_landmarks, guitar_nut_point=None, guitar_body_point=None):
    """
    Convenience wrapper: given a single pose's landmark dict (from
    pose_tracking.detect_pose_landmarks), compute the available angles.
    Guitar angle is only computed if both guitar points are supplied.

    Returns a dict of {metric_name: value_in_degrees}, plus a `notes` list
    of plain-language observations. Does NOT return a pass/fail verdict --
    that judgment is left to the person reading the numbers, consistent
    with this project's approach of reporting facts rather than diagnoses.
    """
    results = {}
    notes = []

    if "left_shoulder" in pose_landmarks and "left_ear" in pose_landmarks:
        cva_left = craniovertebral_angle(pose_landmarks["left_shoulder"], pose_landmarks["left_ear"])
        results["cva_left_deg"] = round(cva_left, 1)
        if cva_left < 50:
            notes.append(f"Left-side CVA ({cva_left:.1f} deg) is below the ~50 deg reference "
                         f"commonly cited for forward head posture in ergonomics research.")

    if "right_shoulder" in pose_landmarks and "right_ear" in pose_landmarks:
        cva_right = craniovertebral_angle(pose_landmarks["right_shoulder"], pose_landmarks["right_ear"])
        results["cva_right_deg"] = round(cva_right, 1)
        if cva_right < 50:
            notes.append(f"Right-side CVA ({cva_right:.1f} deg) is below the ~50 deg reference "
                         f"commonly cited for forward head posture in ergonomics research.")

    if "left_shoulder" in pose_landmarks and "right_shoulder" in pose_landmarks:
        tilt = shoulder_tilt_angle(pose_landmarks["left_shoulder"], pose_landmarks["right_shoulder"])
        results["shoulder_tilt_deg"] = round(tilt, 1)
        if tilt > 5:
            notes.append(f"Shoulder line is tilted {tilt:.1f} deg from level "
                         f"(small amounts are normal; worth noting if this feels persistent or effortful).")

    if guitar_nut_point is not None and guitar_body_point is not None:
        g_angle = guitar_neck_angle(guitar_nut_point, guitar_body_point)
        results["guitar_neck_angle_deg"] = round(g_angle, 1)
        notes.append(f"Guitar neck angle: {g_angle:.1f} deg from horizontal "
                      f"(descriptive only -- correct angle depends on playing style/technique).")

    results["notes"] = notes
    return results
