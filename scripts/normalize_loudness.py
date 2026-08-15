#!/usr/bin/env python3
"""
Equalise level across EVERY clip in the tree -- authentic and synthetic alike.

Why: Piper peak-normalises its output to digital full scale while LibriSpeech
sits around half scale, so before this step peak amplitude alone separated
authentic from Piper at 1.000 balanced accuracy. A model trained on that learns
"loud => Piper", posts an excellent in-domain EER, and collapses the moment it
meets a quiet fake or a loud real recording. Level is a property of the
recording chain, not of whether speech was synthesised, so it must carry no
class information.

The fix has to be applied to BOTH classes identically. Normalising only the
synthetic side would just move the boundary, not remove it.

RMS, not LUFS. RMS is exactly idempotent (normalising twice to the same target
is a no-op), needs no extra dependency, and directly zeroes the axis that was
leaking. LUFS (ITU-R BS.1770 K-weighting) is the better perceptual measure and
would be the right call if the goal were matching listener loudness, but here
the goal is removing a synthetic gain difference, and RMS does that exactly.
Caveat recorded in SOURCES.md: whole-clip RMS includes silence, so clips with
more leading/trailing silence end up with slightly hotter speech.

Idempotent: re-running is a no-op (gains converge to 1.0).

    python scripts/normalize_loudness.py
    python scripts/normalize_loudness.py --dry-run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("data/raw_audio")
LOG = Path("data/raw_audio/loudness_log.json")

# Chosen to sit below every clip's clipping ceiling: with a global max crest
# factor around 17, a 0.05 target puts the loudest peak near 0.85, leaving
# headroom without any limiting. Verified against the actual tree before writing.
TARGET_RMS = 0.05
PEAK_CEILING = 0.99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=float, default=TARGET_RMS)
    args = ap.parse_args()

    clips = sorted(ROOT.rglob("*.wav"))
    if not clips:
        raise SystemExit(f"no wavs under {ROOT}")

    # Pass 1: would this target clip anything? Fail loudly BEFORE writing.
    worst_peak, worst_path, crests = 0.0, None, []
    for p in clips:
        x, _ = sf.read(str(p), dtype="float32")
        rms = float(np.sqrt(np.mean(x**2)))
        if rms <= 0:
            raise ValueError(f"silent clip, refusing to normalise: {p}")
        crest = float(np.max(np.abs(x))) / rms
        crests.append(crest)
        if crest * args.target > worst_peak:
            worst_peak, worst_path = crest * args.target, p
    print(f"{len(clips)} clips | max crest {max(crests):.2f} "
          f"(median {np.median(crests):.2f})")
    print(f"target RMS {args.target} -> worst resulting peak {worst_peak:.4f} "
          f"({worst_path.name})")
    if worst_peak > PEAK_CEILING:
        raise SystemExit(
            f"target {args.target} would drive {worst_path} to {worst_peak:.3f} "
            f"> {PEAK_CEILING}; lower --target to <= {PEAK_CEILING / max(crests):.4f}")
    print("no clip would exceed full scale -- safe to apply\n")

    if args.dry_run:
        print("dry run, nothing written")
        return

    log = {}
    by_gen: dict[str, list] = {}
    for p in clips:
        x, sr = sf.read(str(p), dtype="float32")
        rms_before = float(np.sqrt(np.mean(x**2)))
        gain = args.target / rms_before
        y = x * gain
        peak_after = float(np.max(np.abs(y)))
        if peak_after > PEAK_CEILING:
            raise ValueError(f"unexpected clip on {p}: {peak_after}")
        sf.write(p, y.astype(np.float32), sr, subtype="PCM_16")
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        gen = rel.split("/")[0]
        log[rel] = dict(gain=round(gain, 6), rms_before=round(rms_before, 6),
                        rms_after=round(args.target, 6),
                        peak_after=round(peak_after, 6), sr=sr)
        by_gen.setdefault(gen, []).append(gain)

    LOG.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"normalised {len(log)} clips -> RMS {args.target}")
    print(f"log -> {LOG}\n")
    print(f"{'generator':12s} {'n':>4s} {'gain_min':>9s} {'gain_med':>9s} {'gain_max':>9s}")
    print("-" * 48)
    for gen, gains in sorted(by_gen.items()):
        g = np.array(gains)
        print(f"{gen:12s} {len(g):4d} {g.min():9.3f} {np.median(g):9.3f} {g.max():9.3f}")
    print("\nGain differs systematically by generator -- that is the whole point:")
    print("it is exactly the level offset that was leaking class identity.")


if __name__ == "__main__":
    main()
