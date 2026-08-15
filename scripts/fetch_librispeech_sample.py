#!/usr/bin/env python3
"""
Pull a small, speaker-diverse authentic-speech sample from LibriSpeech dev-clean.

Why this exists: a Track B anti-spoofing set needs a bonafide side sourced from
consented, licensed recordings, and it needs to be split by SPEAKER, not by
utterance. Saving one directory per speaker is what makes that split possible
later -- `make_manifest.py` reads `root/<generator>/<source_id>/*` and uses the
second path component as `source_id`, so speaker becomes the leakage-control
group for free.

Streams the dataset rather than downloading the archive: dev-clean is small but
the parent repo is not, and we only need ~50 clips.

    python scripts/fetch_librispeech_sample.py
    python scripts/fetch_librispeech_sample.py --speakers 10 --per-speaker 5

Re-running is safe: utterances already on disk are skipped, and speakers whose
directory is already full are not re-fetched.
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

REPO = "openslr/librispeech_asr"
CONFIG = "clean"
SPLIT = "validation"

# Keep clips short: long reads waste disk and the artifact signal we care about
# lives in the first few seconds anyway. Bounds are deliberate, not arbitrary.
MIN_DUR_S = 2.0
MAX_DUR_S = 12.0

# Stop scanning eventually even if the stream never yields enough speakers,
# rather than looping forever on a malformed dataset.
MAX_SCAN = 4000

SPEAKER_FIELDS = ("speaker_id", "speaker", "spk_id")
UTT_FIELDS = ("id", "utterance_id", "file", "audio_id")


def pick_field(keys, candidates, what: str) -> str:
    """Resolve a field name against what the dataset ACTUALLY has.

    Assuming LibriSpeech's schema and being wrong produces a directory tree
    that looks right and groups by the wrong thing, which is worse than a crash.
    """
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"no {what} field found; tried {candidates}, dataset has {sorted(keys)}")


def extract_audio(value):
    """Return (float array, sampling_rate) from whatever the Audio feature yields.

    We deliberately load the column with decode=False and decode the raw FLAC
    bytes with libsndfile here. datasets>=4 hands decoding to torchcodec, which
    needs FFmpeg shared libraries present on the system -- an extra native
    dependency to acquire an already-lossless file we then just re-encode.
    The other branches stay so this keeps working if the column arrives decoded.
    """
    if isinstance(value, dict) and value.get("bytes") is not None:
        return sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)

    if isinstance(value, dict) and "array" in value:
        return np.asarray(value["array"]), int(value["sampling_rate"])

    # torchcodec AudioDecoder (datasets >= 4.0)
    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        arr = samples.data
        arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        return arr, int(samples.sample_rate)

    if hasattr(value, "array") and hasattr(value, "sampling_rate"):
        return np.asarray(value.array), int(value.sampling_rate)

    if isinstance(value, dict) and value.get("path"):
        return sf.read(value["path"], dtype="float32", always_2d=False)

    raise TypeError(f"unrecognised audio value of type {type(value)!r}")


def to_mono_frames(arr: np.ndarray) -> np.ndarray:
    """Collapse to a 1-D frame array. torchcodec gives (channels, frames)."""
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # channels-first if the short axis is first
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr.mean(axis=1)
    raise ValueError(f"unexpected audio shape {arr.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/raw_audio")
    ap.add_argument("--speakers", type=int, default=10)
    ap.add_argument("--per-speaker", type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.out_root) / "authentic"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"streaming {REPO} config={CONFIG!r} split={SPLIT!r} ...")
    ds = load_dataset(REPO, CONFIG, split=SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # decode ourselves, see extract_audio

    per_speaker: dict[str, int] = {}
    written = 0
    skipped = 0
    scanned = 0
    spk_field = utt_field = audio_field = None

    def quota_filled() -> bool:
        return (len(per_speaker) >= args.speakers
                and all(v >= args.per_speaker for v in per_speaker.values()))

    for sample in ds:
        # Checked at the TOP so a fully-populated re-run stops as soon as the
        # last speaker is accounted for. Checking only after a successful write
        # means a no-op re-run streams the entire split to do nothing.
        if quota_filled():
            break

        scanned += 1
        if scanned > MAX_SCAN:
            print(f"! hit MAX_SCAN={MAX_SCAN} before filling every speaker")
            break

        if spk_field is None:
            # Confirm the real schema before a single file is written.
            print(f"\nfirst sample.keys() -> {sample.keys()}\n")
            spk_field = pick_field(sample.keys(), SPEAKER_FIELDS, "speaker-id")
            utt_field = pick_field(sample.keys(), UTT_FIELDS, "utterance-id")
            audio_field = pick_field(sample.keys(), ("audio",), "audio")
            print(f"using speaker field={spk_field!r} utterance field={utt_field!r}\n")

        spk = str(sample[spk_field])

        # Speaker already complete (this run or a previous one) -> don't decode.
        if per_speaker.get(spk, 0) >= args.per_speaker:
            continue
        if spk not in per_speaker and len(per_speaker) >= args.speakers:
            continue

        spk_dir = out_root / spk
        if spk not in per_speaker:
            existing = len(list(spk_dir.glob("*.wav"))) if spk_dir.exists() else 0
            per_speaker[spk] = existing
            if existing >= args.per_speaker:
                print(f"  speaker {spk}: already has {existing} clips, skipping")
                continue

        utt = Path(str(sample[utt_field])).stem
        dest = spk_dir / f"{utt}.wav"
        if dest.exists():
            skipped += 1
            per_speaker[spk] = per_speaker.get(spk, 0) + 1
            continue

        arr, sr = extract_audio(sample[audio_field])
        frames = to_mono_frames(arr)
        dur = len(frames) / sr
        if not (MIN_DUR_S <= dur <= MAX_DUR_S):
            continue
        if not np.isfinite(frames).all():
            raise ValueError(f"non-finite samples in {utt}")
        if float(np.max(np.abs(frames))) < 1e-4:
            print(f"  ! {utt} is near-silent, skipping")
            continue

        spk_dir.mkdir(parents=True, exist_ok=True)
        sf.write(dest, frames, sr)  # native rate, no resampling
        per_speaker[spk] = per_speaker.get(spk, 0) + 1
        written += 1

    # ---- summary over what is actually on disk, not what we think we wrote
    clips = sorted(out_root.rglob("*.wav"))
    speakers = sorted({p.parent.name for p in clips})
    total_s = 0.0
    for p in clips:
        info = sf.info(str(p))
        total_s += info.frames / info.samplerate
    total_bytes = sum(p.stat().st_size for p in clips)

    print(f"\nwrote {written} new clips, skipped {skipped} already present "
          f"(scanned {scanned} stream items)")
    print(f"SUMMARY: {len(speakers)} speakers | {len(clips)} clips | "
          f"{total_s / 60:.2f} min total | {total_bytes / 1e6:.2f} MB on disk")


if __name__ == "__main__":
    main()
