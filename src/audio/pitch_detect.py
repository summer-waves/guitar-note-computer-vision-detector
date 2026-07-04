"""
src/audio/pitch_detect.py

Guitar pitch detection via windowed autocorrelation, with octave-error
correction and confidence scoring. Pure numpy/scipy -- no librosa/basic-pitch
dependency required.

Usage as CLI:
    python3 src/audio/pitch_detect.py data/raw_recordings/new_recording_20.wav

Usage as module:
    from src.audio.pitch_detect import analyze_file, collapse_to_notes
    events = analyze_file("path/to.wav")
    notes = collapse_to_notes(events)
"""
import sys
from collections import Counter
import numpy as np
from scipy.io import wavfile

# ---- Config ----
FRAME_MS = 80
HOP_MS = 40

# Standard guitar range: open low E (E2, 82.4Hz) to roughly the 19th-20th fret
# on the high E string (~660-740Hz). We allow a bit of headroom above/below
# for bends, capo use, and alternate tunings, but keep it tight enough to
# reject most transient/noise-driven octave jumps.
MIN_FREQ = 75
MAX_FREQ = 750

SILENCE_RMS_THRESH = 0.01
CONFIDENCE_THRESH = 0.35       # min normalized ACF peak height to accept a pitch
NOTE_MERGE_GAP_S = 0.06
MIN_NOTE_DUR_S = 0.06

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq):
    if freq <= 0:
        return None, None
    A4 = 440.0
    semitones_from_a4 = 12 * np.log2(freq / A4)
    nearest_semitone = round(semitones_from_a4)
    cents_off = (semitones_from_a4 - nearest_semitone) * 100
    midi_num = nearest_semitone + 69
    note_name = NOTE_NAMES[midi_num % 12]
    octave = midi_num // 12 - 1
    return f"{note_name}{octave}", cents_off


def _acf(frame):
    """Normalized autocorrelation via FFT."""
    frame = frame - np.mean(frame)
    windowed = frame * np.hanning(len(frame))
    n = len(windowed)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f = np.fft.rfft(windowed, n=fft_size)
    acf = np.fft.irfft(f * np.conj(f))
    acf = acf[:n]
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def _peak_value_at_lag(acf, lag):
    """Value of the (parabolically interpolated) ACF peak nearest to `lag`."""
    lag = int(round(lag))
    if lag <= 0 or lag >= len(acf) - 1:
        return 0.0, lag
    # snap to local max within a small window
    window = 3
    lo = max(1, lag - window)
    hi = min(len(acf) - 2, lag + window)
    local = acf[lo:hi + 1]
    best = lo + int(np.argmax(local))
    return acf[best], best


def autocorrelation_pitch(frame, sr, min_freq=MIN_FREQ, max_freq=MAX_FREQ):
    """
    Estimate fundamental frequency of a frame via autocorrelation, with
    octave-error correction: if a strong peak exists at 2x the detected
    frequency's lag (i.e. half the period), prefer it, since autocorrelation
    commonly locks onto the wrong octave (usually reporting a subharmonic).
    Returns (freq_hz, confidence) where confidence is the normalized ACF peak height.
    """
    acf = _acf(frame)
    if acf is None:
        return 0.0, 0.0

    min_lag = int(sr / max_freq)
    max_lag = min(int(sr / min_freq), len(acf) - 1)
    if min_lag >= max_lag:
        return 0.0, 0.0

    search = acf[min_lag:max_lag]
    d = np.diff(search)

    # collect all local maxima above a low floor, in lag order
    peaks = []
    for i in range(1, len(d)):
        if d[i - 1] > 0 and d[i] <= 0:
            if search[i] > 0.2:
                peaks.append((min_lag + i, search[i]))

    if not peaks:
        best_i = int(np.argmax(search))
        if search[best_i] < CONFIDENCE_THRESH:
            return 0.0, float(search[best_i]) if len(search) else 0.0
        peaks = [(min_lag + best_i, search[best_i])]

    # peaks are already in increasing lag order (increasing lag = decreasing freq)
    # the FIRST strong peak (shortest lag = highest freq) is usually the true
    # fundamental; later peaks tend to be integer multiples of the period
    # (i.e. subharmonics / lower octaves of the same note). Prefer the
    # shortest-lag peak that clears the confidence threshold.
    candidate_lag, candidate_val = None, None
    for lag, val in peaks:
        if val >= CONFIDENCE_THRESH:
            candidate_lag, candidate_val = lag, val
            break

    if candidate_lag is None:
        # fall back to the strongest peak even if under threshold, but report
        # the low confidence so the caller can decide to discard it
        candidate_lag, candidate_val = max(peaks, key=lambda p: p[1])

    # parabolic interpolation for sub-sample lag accuracy
    lag = candidate_lag
    if 0 < lag < len(acf) - 1:
        y0, y1, y2 = acf[lag - 1], acf[lag], acf[lag + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            lag = lag + shift

    freq = sr / lag
    return freq, float(candidate_val)


def analyze_file(path, min_freq=MIN_FREQ, max_freq=MAX_FREQ, open_strings_only=False):
    """Run frame-by-frame pitch detection over a wav file.

    Args:
        open_strings_only: if True, restricts the search range to 70-360Hz
            (covers all 6 standard open strings with margin for drop tunings).
            This is the fix for octave-doubling errors: if the detector's
            search window can't "see" a candidate lag corresponding to the
            harmonic frequency in the first place, it can't lock onto it by
            mistake. Use this specifically when validating tuning on open
            strings; don't use it for general fretted-note detection, since
            it would clip out legitimately high fretted notes.

    Returns a list of dicts: {time, freq, note, cents, rms, confidence}
    """
    if open_strings_only:
        min_freq, max_freq = 70, 360

    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)

    events = []
    for start in range(0, max(0, len(data) - frame_len), hop_len):
        frame = data[start:start + frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        t = start / sr

        if rms < SILENCE_RMS_THRESH:
            events.append({"time": t, "freq": 0.0, "note": None, "cents": None,
                            "rms": rms, "confidence": 0.0})
            continue

        freq, conf = autocorrelation_pitch(frame, sr, min_freq, max_freq)

        if freq <= 0 or conf < CONFIDENCE_THRESH:
            events.append({"time": t, "freq": 0.0, "note": None, "cents": None,
                            "rms": rms, "confidence": conf})
            continue

        note, cents = freq_to_note(freq)
        events.append({"time": t, "freq": freq, "note": note, "cents": cents,
                        "rms": rms, "confidence": conf})

    return events


def collapse_to_notes(events, merge_gap_s=NOTE_MERGE_GAP_S, min_dur_s=MIN_NOTE_DUR_S):
    """Merge consecutive same-note frames into discrete note events."""
    notes = []
    current = None

    for e in events:
        note = e["note"]
        t = e["time"]

        if note is None:
            if current is not None:
                notes.append(current)
                current = None
            continue

        if current is None:
            current = {"note": note, "start": t, "end": t,
                        "freqs": [e["freq"]], "cents": [e["cents"]],
                        "confidences": [e["confidence"]]}
        elif current["note"] == note and (t - current["end"]) <= merge_gap_s:
            current["end"] = t
            current["freqs"].append(e["freq"])
            current["cents"].append(e["cents"])
            current["confidences"].append(e["confidence"])
        else:
            notes.append(current)
            current = {"note": note, "start": t, "end": t,
                        "freqs": [e["freq"]], "cents": [e["cents"]],
                        "confidences": [e["confidence"]]}

    if current is not None:
        notes.append(current)

    notes = [n for n in notes if (n["end"] - n["start"]) >= min_dur_s]

    for n in notes:
        n["avg_freq"] = float(np.mean(n["freqs"]))
        n["avg_cents"] = float(np.mean(n["cents"]))
        n["avg_confidence"] = float(np.mean(n["confidences"]))
        n["duration"] = n["end"] - n["start"]

    return notes


def summarize_notes(notes, low_conf_thresh=0.5):
    """
    Produce a QC summary dict for a batch of detected notes: how many were
    detected, how confident the detector was overall, how far off standard
    12-TET pitch things landed, and a per-pitch-class breakdown. Useful for
    quickly judging whether a recording/detector run is "good enough" to
    trust, without reading through a raw per-note table.
    """
    if not notes:
        return {"count": 0}

    confs = np.array([n["avg_confidence"] for n in notes])
    cents = np.array([n["avg_cents"] for n in notes])
    durs = np.array([n["duration"] for n in notes])

    low_conf = [n for n in notes if n["avg_confidence"] < low_conf_thresh]

    pitch_class_counts = Counter(n["note"] for n in notes)

    return {
        "count": len(notes),
        "mean_confidence": float(np.mean(confs)),
        "median_confidence": float(np.median(confs)),
        "pct_low_confidence": 100.0 * len(low_conf) / len(notes),
        "mean_cents_off_signed": float(np.mean(cents)),
        "mean_abs_cents_off": float(np.mean(np.abs(cents))),
        "mean_note_duration_s": float(np.mean(durs)),
        "unique_notes_detected": len(pitch_class_counts),
        "most_common_notes": pitch_class_counts.most_common(5),
        "low_confidence_notes": low_conf,
    }


def print_summary(summary):
    if summary["count"] == 0:
        print("No notes detected.")
        return

    print("\n=== Detection Quality Summary ===")
    print(f"Total notes detected:       {summary['count']}")
    print(f"Unique pitch classes seen:  {summary['unique_notes_detected']}")
    print(f"Mean confidence:            {summary['mean_confidence']:.2f}")
    print(f"Median confidence:          {summary['median_confidence']:.2f}")
    print(f"Low-confidence notes:       {summary['pct_low_confidence']:.1f}%")
    print(f"Mean cents off (signed):    {summary['mean_cents_off_signed']:+.1f}c")
    print(f"Mean |cents off|:           {summary['mean_abs_cents_off']:.1f}c")
    print(f"Mean note duration:         {summary['mean_note_duration_s']:.2f}s")
    print("\nMost common notes detected:")
    for note, cnt in summary["most_common_notes"]:
        print(f"  {note}: {cnt}")

    # rough plain-language verdict to help judge "is this good enough"
    verdict_bits = []
    if summary["mean_confidence"] >= 0.75 and summary["pct_low_confidence"] < 15:
        verdict_bits.append("detection confidence looks solid")
    else:
        verdict_bits.append("detection confidence is borderline -- consider re-recording closer to the mic or reducing background noise")

    if summary["mean_abs_cents_off"] <= 15:
        verdict_bits.append("pitch accuracy is tight (within ~15 cents on average)")
    elif summary["mean_abs_cents_off"] <= 30:
        verdict_bits.append("pitch accuracy is acceptable but not tight -- could be genuine tuning drift or estimation noise")
    else:
        verdict_bits.append("pitch accuracy is loose (>30 cents average) -- check guitar tuning or algorithm settings")

    print("\nVerdict: " + "; ".join(verdict_bits) + ".")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pitch_detect.py <audio.wav> [--summary]")
        sys.exit(1)

    path = sys.argv[1]
    show_summary_only = "--summary" in sys.argv
    open_strings_only = "--open-strings-only" in sys.argv

    events = analyze_file(path, open_strings_only=open_strings_only)
    notes = collapse_to_notes(events)

    if not show_summary_only:
        print(f"\nDetected {len(notes)} note(s):\n")
        header = f"{'Time (s)':>10} | {'Dur (s)':>8} | {'Note':>5} | {'Freq (Hz)':>10} | {'Cents off':>10} | {'Conf':>5}"
        print(header)
        print("-" * len(header))
        for n in notes:
            print(f"{n['start']:>10.2f} | {n['duration']:>8.2f} | {n['note']:>5} | "
                  f"{n['avg_freq']:>10.2f} | {n['avg_cents']:>+9.1f}c | {n['avg_confidence']:>5.2f}")

    summary = summarize_notes(notes)
    print_summary(summary)


if __name__ == "__main__":
    main()
