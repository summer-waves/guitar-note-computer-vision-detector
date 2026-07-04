# 🎸 Guitar Note Verifier

## 📌 Overview

A **computer-vision + audio-processing system** that verifies whether a guitarist's
finger placement and played notes match a target reference — combining real-time
pitch detection, fretboard geometry, and hand/pose tracking into one working app.

**🔗 Live app:** https://guitar-note-computer-vision-detector-vkz2o9qvngkl2nrp9pmk52.streamlit.app/

The pipeline:
* Detects played notes from raw audio using **FFT-based autocorrelation pitch tracking** — built from scratch, zero ML-audio dependencies.
* Calibrates a camera's view of a real guitar neck into exact **(string, fret) coordinates**, using the true equal-tempered fret-spacing formula.
* Tracks hand and finger position via **MediaPipe HandLandmarker**, mapping fingertip pixels to fret numbers.
* Tracks body posture via **MediaPipe PoseLandmarker**, computing ergonomics angles (e.g. craniovertebral angle) sourced from real physical-therapy literature.
* Parses and generates **MIDI reference files** from scratch to score played notes against a target sequence, using a hand-built DTW sequence aligner.
* Ships as a working **Streamlit app**, deployed live, with both file-upload and live camera/microphone input.

---

## 📑 Table of Contents

1. [What this demonstrates](#-what-this-demonstrates)
2. [Project structure](#-project-structure)
3. [Architecture](#-architecture)
4. [Stack](#-stack)
5. [Validation: what's actually been tested](#-validation-whats-actually-been-tested)
6. [Real results](#-real-results)
7. [Local app tabs](#-local-app-tabs)
8. [Setup](#-setup)
9. [Usage](#-usage)
10. [Known limitations & roadmap](#-known-limitations--roadmap)

---

## 🎯 What this demonstrates

* Building an audio DSP pipeline from scratch — no librosa, no basic-pitch, just FFT + autocorrelation + confidence scoring
* Writing a **from-scratch MIDI file parser/writer** (no pretty_midi/mido) and a **hand-rolled DTW aligner** (no dtw-python) — all built under a real network-access constraint that turned into a deliberate zero-dependency design choice
* Real camera-geometry math: mapping pixel coordinates to physical fretboard positions using the correct non-linear (equal-tempered) fret spacing, not a naive linear grid
* Applying MediaPipe's hand and pose landmark models to a real, physical, non-trivial tracking problem
* Deploying and debugging a real cloud app end-to-end: dependency conflicts, headless-Linux graphics library chains (`libGL`, `libGLESv2`, `libEGL`), and Windows/PATH environment issues — the unglamorous but genuine engineering work of shipping something real

---

## 📂 Project structure

```text
guitar-note-verifier/
├── data/
│   ├── raw_recordings/       # audio test recordings (.wav)
│   ├── reference_songs/      # MIDI reference files for note-diffing
│   └── vision_samples/       # test photos (fretboard, hand, pose)
├── src/
│   ├── audio/
│   │   └── pitch_detect.py           # FFT-autocorrelation pitch detection
│   ├── reference/
│   │   ├── midi_io.py                # from-scratch MIDI reader/writer
│   │   └── diff.py                   # DTW alignment + note accuracy scoring
│   ├── vision/
│   │   ├── fretboard_calibration.py  # pixel <-> (string, fret) homography
│   │   ├── detect_frets.py           # auto-detect fret markers + fret<->row math
│   │   ├── hand_tracking.py          # MediaPipe hand landmark wrapper
│   │   ├── pose_tracking.py          # MediaPipe pose landmark wrapper
│   │   └── posture_analysis.py       # posture angle math (CVA, shoulder tilt)
│   └── app/
│       └── streamlit_app.py          # deployed multi-tab UI
├── tests/
│   └── test_fretboard_calibration.py # synthetic validation, 6/6 passing
├── packages.txt                      # apt deps for cloud deployment (libgl1, libgles2, libegl1)
├── requirements.txt
└── README.md
```

---

## 🛠 Stack

| Layer | Tool |
|---|---|
| Audio pitch detection | NumPy/SciPy FFT + autocorrelation (from scratch) |
| MIDI I/O | Custom parser/writer (from scratch) |
| Sequence alignment | Custom DTW implementation (from scratch) |
| Fretboard geometry | OpenCV homography + equal-tempered spacing formula |
| Hand tracking | MediaPipe HandLandmarker |
| Pose tracking | MediaPipe PoseLandmarker |
| App / UI | Streamlit (file upload + live camera/mic) |
| Deployment | Streamlit Community Cloud |
| Audio conversion | ffmpeg |

---

## ✅ Validation: what's actually been tested

Every component below was tested against **real recordings and real photos**, not synthetic data alone — and every known limitation is stated plainly rather than glossed over.

| Component | Validated how | Result |
|---|---|---|
| Pitch detection | 3 real guitar recordings (random notes, 5-min chromatic sweep, tuning checks) | 280 notes detected, 82.6% mean confidence; fixed a real octave-doubling bug |
| Fretboard calibration | Real photo of an actual guitar, auto-detected fret markers | **R² = 0.99989**, ~2px mean error across 20 frets |
| Hand tracking | Real photo of an actual hand on the guitar | All 5 fingertips + full skeleton correctly landmarked |
| Pose tracking | Real photos, front and 3/4 angle | All 7 tracked landmarks (ears, nose, shoulders, hips) correctly placed |
| MIDI I/O | Round-trip test (write → read → compare) | Exact match on pitch and timing |
| Fretboard synthetic test | 6 simulated fret/string positions from a skewed camera angle | 6/6 correct |
| Streamlit app | Deployed live on Streamlit Cloud, all 4 tabs | Fully functional after resolving dependency/graphics-library issues |

---

## 📊 Real results

**Audio pitch detection**, run on a real 5-minute guitar recording:
```json
{
  "count": 280,
  "mean_confidence": 0.826,
  "mean_abs_cents_off": 27.6,
  "unique_notes_detected": 36
}
```

**Fretboard calibration**, fit against a real photo:
```
Fitted nut row (px):    22.6
Fitted scale (px):      1339.7
Fit quality (R^2):      0.99989
```

Every fret from 0–20 predicted by this fit lines up almost exactly with the
real fret wires when overlaid back on the photo.

---

## 🖥️ Local app tabs

1. **Audio Pitch Detection** — upload a `.wav` or record live via mic
2. **Fretboard Calibration** — upload/capture a neck photo, see calibration fit
3. **Hand Tracking** — upload/capture a hand photo, see landmark overlay
4. **Finger-to-Fret (Combined)** — two-step calibration + fretting photo → estimated fret per finger

---

## ⚙️ Setup

```bash
pip install numpy scipy opencv-python-headless mediapipe matplotlib streamlit
```

ffmpeg is required for non-WAV audio conversion:
```bash
winget install ffmpeg   # Windows
```

For cloud deployment (Streamlit Cloud or similar headless Linux), `packages.txt` must include:
```
libgl1
libgles2
libegl1
```

---

## ▶️ Usage

```bash
# Pitch detection
python src/audio/pitch_detect.py data/raw_recordings/your_file.wav --summary

# Fretboard calibration from a neck photo
python src/vision/detect_frets.py data/vision_samples/your_neck_photo.jpg

# Hand landmark detection
python src/vision/hand_tracking.py data/vision_samples/your_hand_photo.jpg

# Pose landmark detection
python src/vision/pose_tracking.py data/vision_samples/your_photo.jpg

# Full local/deployed web app
streamlit run src/app/streamlit_app.py
```

---

## 🚀 Known limitations & roadmap

| Limitation | Status | Fix |
|---|---|---|
| No polyphonic pitch detection (chords/strums) | Current | Explore CREPE or basic-pitch if network access allows |
| Finger-to-fret detects fret only, not string | Current | Wire in full 2D homography from `fretboard_calibration.py` |
| No real accuracy number for finger-to-fret yet | Current | Needs one real photo of a finger actually pressed on a fret |
| No valid CVA (posture) reading yet | Current | Needs a true side-profile photo (front-facing photos tested so far aren't valid for this metric) |
| Manual neck cropping required for fret detection | Current | Automate full-photo neck localization |
| Pitch accuracy ~20-30 cents average deviation | Current | Precision ceiling of simple autocorrelation vs. neural pitch trackers |
| MIDI diff scoring untested against real ground truth | Current | Score a real narrated fret-by-fret recording against its true reference |
