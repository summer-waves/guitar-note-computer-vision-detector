"""
src/reference/midi_io.py

Minimal Standard MIDI File (SMF) reader/writer, built from scratch since
mido/pretty_midi aren't installable in this environment. Supports just
enough of the format for our use case: single-track, note-on/note-off
events, one tempo, 4/4-agnostic (we only care about absolute seconds).

This is NOT a general-purpose MIDI library -- it covers format 0 and
format 1 files with simple note on/off pairs, which is what you'll get
from any DAW/notation tool exporting a simple melody or riff.
"""
import struct


DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO_USEC_PER_BEAT = 500000  # 120 BPM


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_var_len(data, pos):
    """Read a MIDI variable-length quantity starting at pos. Returns (value, new_pos)."""
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos


def parse_midi(path):
    """
    Parse a Standard MIDI File into a list of note events.

    Returns a list of dicts: {"pitch": midi_note_num, "start": seconds, "end": seconds}
    sorted by start time. Only note_on (velocity>0) / note_off (or note_on vel=0)
    pairs are tracked; other MIDI events (control changes, etc.) are skipped.
    """
    with open(path, "rb") as f:
        data = f.read()

    pos = 0
    assert data[0:4] == b"MThd", "Not a valid MIDI file (missing MThd header)"
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, n_tracks, division = struct.unpack(">HHH", data[8:8 + header_len])
    pos = 8 + header_len

    if division & 0x8000:
        raise ValueError("SMPTE time division not supported")
    ticks_per_beat = division

    tempo_usec_per_beat = DEFAULT_TEMPO_USEC_PER_BEAT  # updated if a tempo meta event is found
    all_notes = []

    for _ in range(n_tracks):
        assert data[pos:pos + 4] == b"MTrk", "Expected MTrk chunk"
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track_start = pos + 8
        track_end = track_start + track_len
        tpos = track_start

        abs_ticks = 0
        running_status = None
        active_notes = {}  # (channel, pitch) -> start_tick

        # gather tempo changes as (tick, usec_per_beat), applied globally (simple: first one wins per section)
        tempo_map = []

        while tpos < track_end:
            delta, tpos = _read_var_len(data, tpos)
            abs_ticks += delta

            status = data[tpos]
            if status < 0x80:
                # running status: reuse previous status byte, this byte is data
                status = running_status
            else:
                tpos += 1
                running_status = status

            event_type = status & 0xF0
            channel = status & 0x0F

            if status == 0xFF:
                # meta event
                meta_type = data[tpos]
                tpos += 1
                length, tpos = _read_var_len(data, tpos)
                meta_data = data[tpos:tpos + length]
                tpos += length
                if meta_type == 0x51 and length == 3:  # set tempo
                    tempo_usec = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                    tempo_map.append((abs_ticks, tempo_usec))
                # other meta events (track name, end of track, etc.) ignored
                continue

            elif status == 0xF0 or status == 0xF7:
                # sysex event
                length, tpos = _read_var_len(data, tpos)
                tpos += length
                continue

            elif event_type == 0x90:  # note on
                pitch = data[tpos]
                velocity = data[tpos + 1]
                tpos += 2
                if velocity > 0:
                    active_notes[(channel, pitch)] = abs_ticks
                else:
                    # note-on with velocity 0 == note off
                    key = (channel, pitch)
                    if key in active_notes:
                        start_tick = active_notes.pop(key)
                        all_notes.append({"pitch": pitch, "start_tick": start_tick, "end_tick": abs_ticks})

            elif event_type == 0x80:  # note off
                pitch = data[tpos]
                tpos += 2  # skip velocity too
                key = (channel, pitch)
                if key in active_notes:
                    start_tick = active_notes.pop(key)
                    all_notes.append({"pitch": pitch, "start_tick": start_tick, "end_tick": abs_ticks})

            elif event_type in (0xA0, 0xB0, 0xE0):  # poly aftertouch, CC, pitch bend: 2 data bytes
                tpos += 2

            elif event_type == 0xC0 or event_type == 0xD0:  # program change, channel aftertouch: 1 data byte
                tpos += 1

            else:
                # unknown/unsupported status - bail out of this track safely
                break

        if not tempo_map:
            tempo_map = [(0, DEFAULT_TEMPO_USEC_PER_BEAT)]

        # convert ticks -> seconds using the tempo map (piecewise constant tempo)
        def ticks_to_seconds(tick):
            seconds = 0.0
            last_tick = 0
            last_tempo = tempo_map[0][1]
            for (t, tempo) in tempo_map:
                if t >= tick:
                    break
                seconds += (t - last_tick) * (last_tempo / 1_000_000.0) / ticks_per_beat
                last_tick = t
                last_tempo = tempo
            seconds += (tick - last_tick) * (last_tempo / 1_000_000.0) / ticks_per_beat
            return seconds

        for n in all_notes:
            n["start"] = ticks_to_seconds(n.pop("start_tick"))
            n["end"] = ticks_to_seconds(n.pop("end_tick"))

        pos = track_end

    all_notes.sort(key=lambda n: n["start"])
    return all_notes


# ---------------------------------------------------------------------------
# Writing (used to generate simple test/reference MIDI files)
# ---------------------------------------------------------------------------

def _write_var_len(value):
    """Encode an integer as a MIDI variable-length quantity."""
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


def write_simple_midi(notes, path, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                       tempo_usec_per_beat=DEFAULT_TEMPO_USEC_PER_BEAT):
    """
    Write a simple single-track MIDI file from a list of notes.
    `notes`: list of dicts with "pitch" (MIDI note number), "start" (seconds), "end" (seconds)
    """
    events = []  # (abs_tick, event_bytes)

    def seconds_to_ticks(t):
        beats = t / (tempo_usec_per_beat / 1_000_000.0)
        return int(round(beats * ticks_per_beat))

    # tempo meta event at tick 0
    tempo_bytes = tempo_usec_per_beat.to_bytes(3, "big")
    events.append((0, b"\xFF\x51\x03" + tempo_bytes))

    for n in notes:
        start_tick = seconds_to_ticks(n["start"])
        end_tick = seconds_to_ticks(n["end"])
        pitch = n["pitch"]
        velocity = n.get("velocity", 100)
        events.append((start_tick, bytes([0x90, pitch, velocity])))
        events.append((end_tick, bytes([0x80, pitch, 0])))

    events.sort(key=lambda e: e[0])

    track_data = b""
    last_tick = 0
    for (tick, event_bytes) in events:
        delta = tick - last_tick
        track_data += _write_var_len(delta) + event_bytes
        last_tick = tick

    # end of track meta event
    track_data += b"\x00\xFF\x2F\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    track_chunk = b"MTrk" + struct.pack(">I", len(track_data)) + track_data

    with open(path, "wb") as f:
        f.write(header + track_chunk)
