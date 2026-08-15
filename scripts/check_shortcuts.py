#!/usr/bin/env python3
"""
Can a single trivial scalar tell authentic from generated? Run this BEFORE
believing any detector result.

The failure this prevents: a detector that scores brilliantly by reading level,
clip length, or silence ratio rather than a synthesis artifact. Those are
properties of the recording and packaging chain, not of whether speech was
generated, and a model leaning on them collapses on the first out-of-domain
batch. If any axis here sits far from 0.5, the dataset is telling the model the
answer and the eval number is fiction.

Balanced accuracy of the best single threshold on each axis, per generator:
0.5 means the axis carries no class information, 1.0 means it fully determines
the class.

    python scripts/check_shortcuts.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("data/raw_audio")
AUTHENTIC = "authentic"
FRAME_MS = 20
SILENCE_REL = 0.1  # frame counts as silent below 10% of the clip's own RMS


def silence_fraction(x: np.ndarray, sr: int) -> float:
    """Scale-invariant: relative to the clip's own RMS, so gain cannot change it."""
    n = max(1, int(sr * FRAME_MS / 1000))
    trimmed = x[: len(x) - len(x) % n]
    if trimmed.size == 0:
        return 0.0
    frames = trimmed.reshape(-1, n)
    frms = np.sqrt(np.mean(frames**2, axis=1))
    clip_rms = float(np.sqrt(np.mean(x**2)))
    if clip_rms <= 0:
        return 1.0
    return float(np.mean(frms < SILENCE_REL * clip_rms))


def band_fraction(x: np.ndarray, sr: int, lo: float = 7000, hi: float = 8000) -> float:
    """Near-Nyquist energy fraction. Catches BANDWIDTH provenance: an authentic
    corpus with upstream lossy encoding rolls off its top band, while TTS
    synthesizes energy to Nyquist -- which once separated our classes at 0.91
    before it was found by hand. This axis makes that check automatic."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    tot = spec.sum()
    return float(spec[(freqs >= lo) & (freqs < hi)].sum() / tot) if tot > 0 else 0.0


def features(path: Path) -> dict:
    x, sr = sf.read(str(path), dtype="float32")
    return dict(
        peak=float(np.max(np.abs(x))),
        rms=float(np.sqrt(np.mean(x**2))),
        duration=len(x) / sr,
        silence=silence_fraction(x, sr),
        hf7_8k=band_fraction(x, sr),
    )


def best_balanced_acc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Best single-threshold balanced accuracy, either polarity."""
    lo, hi = min(neg.min(), pos.min()), max(neg.max(), pos.max())
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return 0.5
    best = 0.5
    for t in np.linspace(lo, hi, 801):
        acc = (np.mean(neg < t) + np.mean(pos >= t)) / 2
        best = max(best, acc, 1 - acc)
    return float(best)


def main() -> None:
    gens = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and any(p.rglob("*.wav")))
    data = {}
    for g in gens:
        rows = [features(p) for p in sorted((ROOT / g).rglob("*.wav"))]
        data[g] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        print(f"{g:12s} n={len(rows)}")

    axes = ["peak", "rms", "duration", "silence", "hf7_8k"]

    print("\n" + "=" * 72)
    print("DISTRIBUTIONS (median [min, max])")
    print("=" * 72)
    print(f"{'generator':12s} " + "".join(f"{a:>22s}" for a in axes))
    for g in gens:
        cells = []
        for a in axes:
            v = data[g][a]
            cells.append(f"{np.median(v):7.3f} [{v.min():.2f},{v.max():.2f}]".rjust(22))
        print(f"{g:12s} " + "".join(cells))

    print("\n" + "=" * 72)
    print("SEPARABILITY -- best single-threshold balanced accuracy vs authentic")
    print("0.50 = no information (goal)   1.00 = axis fully determines the class")
    print("=" * 72)
    print(f"{'generator':12s} " + "".join(f"{a:>11s}" for a in axes))
    worst = (0.0, None, None)
    for g in gens:
        if g == AUTHENTIC:
            continue
        cells = []
        for a in axes:
            acc = best_balanced_acc(data[AUTHENTIC][a], data[g][a])
            flag = "" if acc < 0.75 else ("*" if acc < 0.9 else "!")
            cells.append(f"{acc:10.3f}{flag}")
            if acc > worst[0]:
                worst = (acc, g, a)
        print(f"{g:12s} " + "".join(cells))

    print("\n  blank = below 0.75   * = 0.75-0.90 (leaky)   ! = >0.90 (shortcut)")
    print(f"\nworst axis: {worst[1]} / {worst[2]} at {worst[0]:.3f}")
    if worst[0] >= 0.9:
        print("  -> a detector can pass by reading this alone. Fix before training.")
    elif worst[0] >= 0.75:
        print("  -> leaky but not decisive; note it and watch it in ablations.")
    else:
        print("  -> no trivial scalar separates the classes.")


if __name__ == "__main__":
    main()
