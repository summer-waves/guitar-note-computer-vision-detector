# Guitar Note Verifier

A computer-vision + audio-processing system for verifying whether a guitarist's
finger placement and played notes match a target reference — built as a
portfolio project to demonstrate ML engineering across audio DSP, computer
vision, and systems design.

**Author:** Marco
**Status:** Full pipeline running end-to-end locally via a working Streamlit
app (all four tabs functional on Windows: audio pitch detection, fretboard
calibration, hand tracking, and a combined finger-to-fret estimator). Core
audio and vision modules are individually tested against real recordings/
photos. The one remaining gap: the combined finger-to-fret tab has been
verified to run correctly end-to-end, but not yet against a photo where a
finger is actually pressed on a fret (see "Known limitations" below).

---

## Why this project

Verifying whether a guitarist is playing the "correct" note requires combining
two independent signals:

1. **What sound came out** (audio pitch detection) — the ground truth for
   "which note was actually produced."
2. **How it was physically played** (computer vision on hand/fretboard) —
   useful for form-checking and diagnosing *why* a note might be wrong
   (wrong fret vs. wrong string vs. muted string).

Relying on vision alone can't reliably distinguish pitch (cameras can't see
string vibration frequency at normal frame rates), and audio alone can't
tell you *how* a note was played. This project builds both halves as
independently testable modules, with the goal of eventually fusing them.

A deliberate constraint on this project: **every module is built with zero
ML-library dependencies beyond numpy/scipy/opencv** (no librosa, no
basic-pitch, no mido/pretty_midi, no dtw-python). This was originally a
practical necessity (no network access in the dev sandbox to install those
packages), but it turned into a useful demonstration of DSP/algorithm
fundamentals: FFT-based autocorrelation pitch detection, a from-scratch MIDI
file parser/writer, a hand-rolled DTW sequence aligner, and closed-form
fretboard geometry (equal-tempered fret spacing).

---

## Architecture

```
guitar-note-verifier/
├── data/
│   ├── raw_recordings/       # audio test recordings (.wav, converted from .m4a)
│   ├── reference_songs/      # MIDI reference files for note-diffing
│   └── vision_samples/       # test photos (fretboard, hand)
├── src/
│   ├── audio/
│   │   └── pitch_detect.py       # FFT-autocorrelation pitch detection
│   ├── reference/
│   │   ├── midi_io.py            # from-scratch MIDI reader/writer
│   │   └── diff.py               # DTW alignment + note accuracy scoring
│   ├── vision/
│   │   ├── fretboard_calibration.py  # pixel <-> (string, fret) homography
│   │   ├── detect_frets.py           # auto-detect fret markers + fret<->row math
│   │   └── hand_tracking.py          # MediaPipe hand landmark wrapper
│   └── app/
│       └── streamlit_app.py      # working local UI: 4 tabs, file upload + live camera/mic
├── tests/
│   └── test_fretboard_calibration.py  # synthetic validation, 6/6 passing
└── notebooks/                    # exploratory analysis / plotting scripts
```

---

## What's built and verified

### 1. Audio pitch detection (`src/audio/pitch_detect.py`)

Frame-by-frame fundamental frequency estimation via windowed, FFT-based
autocorrelation, with:
- Sub-sample lag interpolation (parabolic) for finer pitch resolution
- Confidence scoring based on normalized autocorrelation peak height
- An `open_strings_only` mode that restricts the search range to 70-360Hz,
  which fixes a real octave-doubling bug discovered during testing (the
  detector was occasionally locking onto a string's harmonic overtone
  instead of its fundamental, especially on weak/quiet plucks)
- A `summarize_notes()` QC report (confidence stats, cents-off accuracy,
  note distribution) for quickly judging whether a recording is usable

**Validated against:** three real guitar recordings (random notes, a 5-minute
full-fretboard chromatic sweep, and dedicated tuning-check clips). Confirmed
via visual waveform/pitch overlays that detected notes track real pluck
onsets accurately, and via automated cross-checking that fixing the octave
bug eliminated clearly-wrong detections (e.g. 440Hz phantom notes that were
actually harmonics of a real 220Hz string).

**Known limitation:** monophonic only — chords/strums produce unreliable
results since the algorithm assumes one dominant pitch per frame. Documented
rather than silently mishandled.

### 2. MIDI I/O (`src/reference/midi_io.py`)

A minimal Standard MIDI File parser and writer, supporting tempo maps,
running status, and format 0/1 files. Built from scratch since `pretty_midi`
wasn't installable in the dev environment.

**Validated via:** exact round-trip test (write a 13-note chromatic scale,
read it back, confirm identical pitches and timings).

### 3. Note diffing (`src/reference/diff.py`)

Compares a detected note sequence against a MIDI reference using a
from-scratch O(n·m) DTW implementation (since `dtw-python` wasn't available),
producing per-note correct/incorrect scoring with timing offsets.

**Validated via:** end-to-end test against a real recording segment scored
against a synthetic reference, confirming the alignment and scoring logic
executes correctly. (Note: real-world accuracy numbers from this test are
not meaningful yet, since building an accurate reference requires knowing
exactly what was played — see Next Steps.)

### 4. Fretboard calibration (`src/vision/fretboard_calibration.py`)

Maps a pixel coordinate from a camera photo to a (string, fret) position via
homography, using the correct **equal-tempered, non-linear** fret-spacing
formula (frets get closer together toward the body — this is physically
correct, not a simplification).

**Validated via:**
- Synthetic test: 6/6 fret+string test cases correctly recovered from a
  simulated skewed camera angle (`tests/test_fretboard_calibration.py`)
- **Real photo test:** automatically detected the position-marker inlays
  (frets 3, 5, 7, 9, 12, 15, 17) on an actual guitar photo and fit them
  against the theoretical fret-spacing model — **R² = 0.99989**, mean
  error of 2 pixels. The fitted model correctly predicts every fret
  position (0-20) when overlaid back on the photo.

### 5. Automated fret detection (`src/vision/detect_frets.py`)

Wraps the calibration validation above into a reusable pipeline: given a
cropped neck photo, automatically detects marker dots (Hough circle
detection) and fits the nut position + scale.

**Known limitation:** expects a photo already roughly cropped to just the
neck, and assumes the marker sequence starts from fret 3 (won't handle a
photo that starts mid-neck). Full-photo neck detection isn't automated yet.

### 6. Hand landmark tracking (`src/vision/hand_tracking.py`)

Wraps MediaPipe's HandLandmarker to detect 21 hand landmarks (including all
5 fingertips) per detected hand.

**Validated via:** real test photo — confirmed visually that all 5
fingertip landmarks land accurately on the actual fingertips, with correctly
connected finger skeleton lines (no crossed/misassigned joints).

**Known limitation:** tested only on a hand resting near the guitar body,
not yet on an actual fretting position (fingers pressed onto the fretboard).
The tracking model itself is confirmed to work on this camera/lighting
setup; combining it with fretboard_calibration.py for a live "which fret is
this finger on" readout is the next integration step.

---

### 7. Local Streamlit app (`src/app/streamlit_app.py`)

A working local web UI wrapping all the modules above into four tabs:

- **Audio Pitch Detection** — upload a `.wav` or record live via microphone
  (`st.audio_input`), see detected notes and the QC summary
- **Fretboard Calibration** — upload a neck photo or capture via webcam
  (`st.camera_input`), see the fitted calibration and fit quality
- **Hand Tracking** — upload/capture a hand photo, see landmark overlay
- **Finger-to-Fret (Combined)** — two-step flow: calibrate against a clean
  neck photo, then upload/capture a fretting photo to get an estimated
  fret number per fingertip

**Validated via:** confirmed running successfully end-to-end on a real
Windows machine (not just this dev sandbox), reproducing near-identical
audio pitch detection results (280 notes, ~83% mean confidence) to the
original sandbox run on the same file — real cross-environment validation.
All four tabs execute without errors on real uploaded photos/recordings.

**Setup gotchas hit and resolved along the way** (worth documenting since
they're common real-world friction points):
- Windows PATH doesn't update in an already-open terminal/IDE after
  installing a new CLI tool (ffmpeg) — needs a full application restart,
  not just a new terminal tab
- A file can have a `.wav` extension while actually containing AAC-encoded
  (m4a) audio internally if only renamed rather than properly converted --
  ffmpeg reads actual file contents regardless of extension, so pointing it
  at the mislabeled file still worked correctly once given the right path
- Installing mediapipe upgraded numpy to a version incompatible with the
  already-installed pandas build, breaking Streamlit's dataframe rendering
  with a binary-incompatibility error until pandas was upgraded to match

### Known limitation on the combined tab specifically

The fret-number math itself is verified correct (round-trip tested against
all frets 0-20 using real calibration numbers from an actual guitar photo --
every fret recovers exactly). However, the one real test run so far used a
fretting-photo where the hand was resting near the sound hole rather than
actually pressed on a fret, so the output (every fingertip reporting the
maximum clamp value of fret 24) reflects that mismatch rather than a real
reading. **Getting one genuine photo of a finger actually pressed on a
specific fret is the last step needed for a real accuracy number** -- the
pipeline is ready for it; we just haven't fed it the right input yet.

Separately: this tab estimates the **fret** number only (vertical position
along the neck). It does not yet determine which **string** a finger is on,
since that requires the fuller 2D homography in `fretboard_calibration.py`
rather than the simpler 1D nut/scale fit `detect_frets.py` currently
produces.

### 8. Posture tracking (`src/vision/pose_tracking.py`, `src/vision/posture_analysis.py`)

An extension beyond finger/note verification: tracking body posture while
playing, using MediaPipe's Pose Landmarker (same pattern as hand tracking,
applied to the whole upper body -- nose, ears, shoulders, hips).

`posture_analysis.py` computes descriptive angles from those landmarks:
- **Craniovertebral angle (CVA)** -- a real metric from ergonomics/physical
  therapy research, used as a general reference point for forward head
  posture. Deliberately implemented with no fixed "correct" verdict, and
  documented with the caveat that researchers don't agree on one universal
  cutoff (commonly cited figures range from ~48-52 degrees).
- **Shoulder tilt** -- simple levelness check.
- **Guitar neck angle** -- purely descriptive (correct angle is a matter of
  playing style/technique, not a single right answer).

**Validated via:** landmark detection confirmed accurate against two real
photos (all 7 tracked points visually verified to land correctly on ears,
shoulders, hips, nose).

**Known limitation -- important:** the CVA metric specifically requires a
true side-profile photo to be clinically meaningful (it measures forward/
backward head position, which only shows up from the side). Both real
test photos taken so far were front-facing/frontal-ish, so while the
landmark detection and angle math both run correctly, **no valid CVA
reading has been produced yet.** Getting one genuine side-profile photo
is needed before this metric can be trusted -- this is flagged directly
in the code's docstring as well, so the limitation travels with the code,
not just this document.

---

## Known limitations (honest accounting)

- **No polyphonic pitch detection.** Chords/strums aren't reliably handled.
- **Manual neck cropping required** for `detect_frets.py` — no full-photo
  automatic neck localization yet.
- **Combined finger-to-fret tab reports fret only, not string** — see above.
- **No real accuracy number yet for finger-to-fret detection** — the one
  test run used a non-fretting hand position (see above); pipeline is ready,
  just needs the right input photo.
- **No valid CVA (posture) reading yet** — requires a true side-profile
  photo, which hasn't been captured; landmark detection and angle math are
  both confirmed working, just not yet fed the right input (same pattern
  as the finger-to-fret gap above).
- **Pitch accuracy ceiling:** even after fixing the octave-doubling bug,
  real recordings show ~20-30 cents average deviation from equal temperament
  — likely a mix of genuine tuning drift and the inherent precision limit of
  a simple autocorrelation approach (vs. more advanced methods like CREPE).
- **Live camera/mic widgets (`st.camera_input`, `st.audio_input`) are
  confirmed working** on the developer's machine, but were written without
  the ability to test them in the dev sandbox (no camera/mic there) — flagging
  this so it's clear that verification happened after the fact, locally.

---

## Next steps

1. Capture one real "finger actually pressed on a specific fret" photo and
   re-run the Finger-to-Fret tab to get the first genuine accuracy reading
   (the pipeline is fully built and tested for this; only the right input
   photo is missing).
2. Extend the combined tab to determine **string**, not just fret, by
   wiring in the full 2D homography from `fretboard_calibration.py`.
3. Capture one genuine side-profile photo to get the first valid CVA
   posture reading (same "pipeline ready, right input still needed"
   situation as item 1).
3. Extend `src/reference/diff.py` scoring to real, deliberately-played test
   recordings with a known ground-truth reference (e.g. a narrated
   fret-by-fret recording), rather than only synthetic references.
4. Consider CREPE or a similar neural pitch tracker if/when network access
   allows installing it, to reduce the ~20-30 cent detection noise floor.
5. Automate full-photo neck localization (currently requires manual
   cropping) so `detect_frets.py` works on an unprocessed camera frame.

---

## Setup

```bash
pip install numpy scipy opencv-python mediapipe matplotlib streamlit
```

ffmpeg is required for converting non-WAV audio (`.m4a`, etc.) — install via
`winget install ffmpeg` on Windows (restart your terminal/IDE fully after
installing, since PATH changes don't apply to already-open sessions).

## Usage examples

```bash
# Pitch detection on a recording
python src/audio/pitch_detect.py data/raw_recordings/your_file.wav --summary

# Same, restricted to open-string range for tuning checks
python src/audio/pitch_detect.py data/raw_recordings/your_file.wav --summary --open-strings-only

# Fretboard calibration from a neck photo
python src/vision/detect_frets.py data/vision_samples/your_neck_photo.jpg

# Hand landmark detection
python src/vision/hand_tracking.py data/vision_samples/your_hand_photo.jpg

# Full local web app (all 4 tabs, file upload + live camera/mic)
streamlit run src/app/streamlit_app.py
```
