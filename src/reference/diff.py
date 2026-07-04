"""
src/reference/diff.py

Compares detected notes (from src.audio.pitch_detect) against a reference
note sequence (from a MIDI file, via src.reference.midi_io) and scores
pitch + timing accuracy.

Uses a from-scratch DTW (dynamic time warping) implementation to align the
two sequences, since a human player will never hit the reference tempo
exactly -- direct index-to-index comparison would be too fragile.
"""
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_num_to_note_name(midi_num):
    note_name = NOTE_NAMES[midi_num % 12]
    octave = midi_num // 12 - 1
    return f"{note_name}{octave}"


def note_name_to_pitch_class(note_name):
    """'A#3' -> pitch class int 0-11 (ignoring octave), for chroma-level comparisons."""
    for i, n in enumerate(NOTE_NAMES):
        if note_name.startswith(n):
            return i
    return None


def note_name_to_midi_num(note_name):
    """'A#3' -> MIDI note number."""
    # find the longest matching note prefix (handles '#' correctly, e.g. 'C#' vs 'C')
    best = None
    for i, n in enumerate(NOTE_NAMES):
        if note_name.startswith(n):
            if best is None or len(n) > len(NOTE_NAMES[best]):
                best = i
    octave = int(note_name[len(NOTE_NAMES[best]):])
    return (octave + 1) * 12 + best


def _pitch_distance(midi_a, midi_b):
    """Distance between two MIDI pitches in semitones (absolute)."""
    return abs(midi_a - midi_b)


def dtw_align(seq_a_pitches, seq_b_pitches):
    """
    Basic O(n*m) DTW alignment between two sequences of MIDI pitch numbers.
    Returns the optimal alignment as a list of (i, j) index pairs (may include
    repeats on either side, standard DTW behavior) and the total cost.
    """
    n, m = len(seq_a_pitches), len(seq_b_pitches)
    if n == 0 or m == 0:
        return [], float("inf")

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _pitch_distance(seq_a_pitches[i - 1], seq_b_pitches[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(choices, key=lambda c: c[0])
    path.reverse()

    return path, cost[n, m]


def score_against_reference(detected_notes, reference_notes, tempo_tolerant=True):
    """
    detected_notes: output of pitch_detect.collapse_to_notes() -- dicts with
        'note' (name string), 'start', 'end', etc.
    reference_notes: output of midi_io.parse_midi() -- dicts with
        'pitch' (MIDI num), 'start', 'end'.

    Returns a dict with per-note match results and an overall accuracy score.
    """
    if not detected_notes or not reference_notes:
        return {"matches": [], "accuracy": 0.0, "note_count_detected": len(detected_notes),
                "note_count_reference": len(reference_notes)}

    detected_pitches = [note_name_to_midi_num(n["note"]) for n in detected_notes]
    reference_pitches = [n["pitch"] for n in reference_notes]

    path, total_cost = dtw_align(detected_pitches, reference_pitches)

    matches = []
    for (i, j) in path:
        det = detected_notes[i]
        ref = reference_notes[j]
        det_pitch = detected_pitches[i]
        ref_pitch = reference_pitches[j]
        semitone_error = det_pitch - ref_pitch
        correct = (semitone_error == 0)
        # timing offset only meaningful if not tempo-warping; report as informational
        timing_offset = det["start"] - ref["start"]

        matches.append({
            "detected_note": det["note"],
            "reference_note": midi_num_to_note_name(ref_pitch),
            "semitone_error": int(semitone_error),
            "correct": correct,
            "detected_time": det["start"],
            "reference_time": ref["start"],
            "timing_offset_s": timing_offset,
        })

    # de-duplicate repeated alignments (DTW can map one reference note to
    # several detected frames if the player held/rearticulated it); collapse
    # by reference index, keeping the first match for scoring purposes
    seen_ref = set()
    unique_matches = []
    for m in matches:
        key = m["reference_time"]
        if key in seen_ref:
            continue
        seen_ref.add(key)
        unique_matches.append(m)

    n_correct = sum(1 for m in unique_matches if m["correct"])
    accuracy = n_correct / len(unique_matches) if unique_matches else 0.0

    return {
        "matches": unique_matches,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(unique_matches),
        "note_count_detected": len(detected_notes),
        "note_count_reference": len(reference_notes),
        "dtw_cost": total_cost,
    }


def print_score_report(result):
    print(f"\n=== Note Accuracy Report ===")
    print(f"Detected notes: {result['note_count_detected']} | Reference notes: {result['note_count_reference']}")
    if result.get("n_total", 0) == 0:
        print("No alignable matches.")
        return
    print(f"Correct: {result['n_correct']} / {result['n_total']}  ({result['accuracy']*100:.1f}%)\n")
    print(f"{'Ref Note':>9} | {'You Played':>10} | {'Semitone Err':>12} | {'Timing Off (s)':>14} | Result")
    print("-" * 65)
    for m in result["matches"]:
        status = "OK" if m["correct"] else "WRONG"
        print(f"{m['reference_note']:>9} | {m['detected_note']:>10} | {m['semitone_error']:>+12d} | "
              f"{m['timing_offset_s']:>+14.2f} | {status}")
