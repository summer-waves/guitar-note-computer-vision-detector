"""
src/app/streamlit_app.py

STATUS: Updated with live camera/audio capture and combined finger-to-fret
detection. The camera/audio widgets below have NOT been tested by Claude
(no camera/mic in the dev sandbox) -- test these locally and report back
if anything misbehaves.

Run with:
    pip install streamlit
    streamlit run src/app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import tempfile
import subprocess
import cv2
import numpy as np
from scipy.io import wavfile

from src.audio.pitch_detect import analyze_file, collapse_to_notes, summarize_notes
from src.vision.detect_frets import calibrate_from_neck_photo, estimate_fret_from_row
from src.vision.hand_tracking import detect_hand_landmarks, draw_landmarks, FINGERTIP_IDS

st.set_page_config(page_title="Guitar Note Verifier", layout="wide")
st.title("Guitar Note Verifier")


def get_audio_path(uploaded_or_recorded):
    """
    Save a Streamlit file-like object (from file_uploader or audio_input) to
    a temp .wav path, converting via ffmpeg if scipy can't read it directly
    (st.audio_input's browser-recorded format isn't guaranteed to be a plain
    PCM WAV that scipy.io.wavfile understands).
    """
    raw_bytes = uploaded_or_recorded.read()
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp.write(raw_bytes)
        raw_path = tmp.name

    try:
        wavfile.read(raw_path)
        return raw_path  # already a plain WAV scipy can read
    except Exception:
        pass

    # fall back to ffmpeg conversion
    converted_path = raw_path + "_converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ar", "44100", "-ac", "1", converted_path],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode(errors='ignore')}")
    return converted_path


tab_audio, tab_neck, tab_hand, tab_combined = st.tabs(
    ["Audio Pitch Detection", "Fretboard Calibration", "Hand Tracking", "Finger-to-Fret (Combined)"]
)

# ---------------------------------------------------------------------------
with tab_audio:
    st.header("Audio Pitch Detection")

    audio_source = st.radio("Input method", ["Upload file", "Record live"], key="audio_source", horizontal=True)
    if audio_source == "Upload file":
        audio_input = st.file_uploader("Upload a .wav recording", type=["wav"])
    else:
        audio_input = st.audio_input("Record a guitar clip")

    open_strings_only = st.checkbox("Restrict to open-string range (70-360Hz) -- use for tuning checks")

    if audio_input is not None:
        try:
            wav_path = get_audio_path(audio_input)
        except Exception as e:
            st.error(f"Could not read/convert audio: {e}")
            wav_path = None

        if wav_path:
            events = analyze_file(wav_path, open_strings_only=open_strings_only)
            notes = collapse_to_notes(events)
            summary = summarize_notes(notes)

            st.subheader("Detected Notes")
            st.dataframe([
                {"Time (s)": round(n["start"], 2), "Note": n["note"],
                 "Freq (Hz)": round(n["avg_freq"], 1), "Cents off": round(n["avg_cents"], 1),
                 "Confidence": round(n["avg_confidence"], 2)}
                for n in notes
            ])

            st.subheader("Summary")
            st.json(summary)

# ---------------------------------------------------------------------------
with tab_neck:
    st.header("Fretboard Calibration")

    neck_source = st.radio("Input method", ["Upload file", "Use camera"], key="neck_source", horizontal=True)
    if neck_source == "Upload file":
        neck_photo = st.file_uploader("Upload a cropped neck photo", type=["jpg", "jpeg", "png"], key="neck")
    else:
        neck_photo = st.camera_input("Capture the neck", key="neck_cam")

    if neck_photo is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(neck_photo.getvalue())
            tmp_path = tmp.name

        result = calibrate_from_neck_photo(tmp_path)
        if result is None:
            st.warning("Not enough fret markers detected. Try a clearer, closer photo of the neck.")
        else:
            st.success(f"Calibration fit quality (R^2): {result['r_squared']:.5f}")
            st.json({"row_nut": result["row_nut"], "scale_px": result["scale_px"],
                     "fret_to_row": result["fret_to_row"]})

# ---------------------------------------------------------------------------
with tab_hand:
    st.header("Hand Landmark Tracking")

    hand_source = st.radio("Input method", ["Upload file", "Use camera"], key="hand_source", horizontal=True)
    if hand_source == "Upload file":
        hand_photo = st.file_uploader("Upload a hand/fretting photo", type=["jpg", "jpeg", "png"], key="hand")
    else:
        hand_photo = st.camera_input("Capture your hand", key="hand_cam")

    if hand_photo is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(hand_photo.getvalue())
            tmp_path = tmp.name

        hands_pixels, img_bgr = detect_hand_landmarks(tmp_path)
        st.write(f"Detected {len(hands_pixels)} hand(s)")

        if hands_pixels:
            vis = draw_landmarks(img_bgr, hands_pixels)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption="Detected hand landmarks")

# ---------------------------------------------------------------------------
with tab_combined:
    st.header("Finger-to-Fret (Combined)")
    st.caption(
        "Two-step process: first calibrate against a clean neck photo (no hand), "
        "then upload/capture a photo of your hand fretting a note. Both photos "
        "should be from roughly the same camera position for accuracy. "
        "Note: this estimates the FRET number from vertical position only -- "
        "it does not yet determine which STRING, since that needs a full 2D "
        "homography (see fretboard_calibration.py) rather than this simpler fit."
    )

    st.subheader("Step 1: Calibration photo (clean neck)")
    calib_source = st.radio("Input method", ["Upload file", "Use camera"], key="calib_source", horizontal=True)
    if calib_source == "Upload file":
        calib_photo = st.file_uploader("Upload a clean neck photo", type=["jpg", "jpeg", "png"], key="calib_upload")
    else:
        calib_photo = st.camera_input("Capture the clean neck", key="calib_cam")

    calibration = None
    if calib_photo is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(calib_photo.getvalue())
            calib_path = tmp.name
        calibration = calibrate_from_neck_photo(calib_path)
        if calibration is None:
            st.warning("Not enough fret markers detected in the calibration photo.")
        else:
            st.success(f"Calibrated (R^2={calibration['r_squared']:.4f}): "
                       f"row_nut={calibration['row_nut']:.1f}, scale_px={calibration['scale_px']:.1f}")

    st.subheader("Step 2: Fretting photo (hand on neck)")
    finger_source = st.radio("Input method", ["Upload file", "Use camera"], key="finger_source", horizontal=True)
    if finger_source == "Upload file":
        finger_photo = st.file_uploader("Upload a fretting photo", type=["jpg", "jpeg", "png"], key="finger_upload")
    else:
        finger_photo = st.camera_input("Capture your fretting hand", key="finger_cam")

    if finger_photo is not None and calibration is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(finger_photo.getvalue())
            finger_path = tmp.name

        hands_pixels, img_bgr = detect_hand_landmarks(finger_path)

        if not hands_pixels:
            st.warning("No hand detected in the fretting photo.")
        else:
            vis = draw_landmarks(img_bgr, hands_pixels)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption="Detected hand + estimated frets below")

            st.subheader("Estimated fret per fingertip")
            pixels = hands_pixels[0]
            rows = []
            for name, idx in FINGERTIP_IDS.items():
                x, y = pixels[idx]
                fret = estimate_fret_from_row(y, calibration["row_nut"], calibration["scale_px"])
                rows.append({"Finger": name, "Pixel Y": round(y, 1), "Estimated Fret": fret})
            st.dataframe(rows)
    elif finger_photo is not None and calibration is None:
        st.info("Complete Step 1 (calibration) first, or Step 1's photo didn't calibrate successfully.")