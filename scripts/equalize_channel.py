#!/usr/bin/env python3
"""
Apply the authentic corpus's high-frequency rolloff to the SYNTHETIC side.

The confound this removes: LibriSpeech is MP3-lineage (LibriVox), so the
authentic clips carry an upstream lossy-codec rolloff in the top of the band
(7-8 kHz energy fraction, median 1.93e-04). The TTS engines synthesize energy
all the way to Nyquist (kokoro 3.09e-03, piper 8.47e-04), so near-Nyquist
energy alone separated the classes at 0.91/0.71 balanced accuracy and the
layer-0 wav2vec2 probe read that corpus fingerprint as "fake". A 16k->24k->16k
round-trip on authentic was measured first and moved the gap by -7.8% (i.e.
widened it): resampling cannot create energy the source never had. The
workable direction is attenuating the synthetic side.

Why this is NOT a violation of "identical processing for both classes": the
authentic side ALREADY passed through this channel upstream, before we ever
touched it. Passing only the synthetic side through an approximation of the
same channel is channel matching -- it moves the classes onto the same
distribution instead of stamping a class-correlated difference on them.

Filter: RBJ audio-EQ-cookbook high-shelf CUT, f0 = 6.5 kHz, gain = -12 dB,
S = 1, single pass. -12 dB is the nearest standard EQ value to the coarse
kokoro/authentic power ratio (~16x ~= 12 dB) read once from the diagnostic --
chosen a priori, not iterated against the post-fix measurement. Erasing the
gap exactly would be overfitting the fix to one statistic; the post-fix number
is REPORTED as whatever it comes out to be, not tuned to 0.5.

Idempotent via data/raw_audio/channel_eq_log.json: a clip already logged is
skipped, so re-running cannot double-apply the shelf (-24 dB).

    python scripts/equalize_channel.py

Run scripts/normalize_loudness.py afterwards: the shelf removes a little
energy, and loudness was equalised tree-wide before this step.
"""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

ROOT = Path("data/raw_audio")
SYNTH_GENERATORS = ("kokoro", "piper")
LOG = ROOT / "channel_eq_log.json"

F0_HZ = 6500.0
GAIN_DB = -12.0
SHELF_S = 1.0


def high_shelf_coeffs(fs: float, f0: float = F0_HZ, gain_db: float = GAIN_DB,
                      s: float = SHELF_S) -> tuple[np.ndarray, np.ndarray]:
    """RBJ audio-EQ-cookbook high-shelf biquad (b, a), normalised to a0=1."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cosw, sinw = math.cos(w0), math.sin(w0)
    alpha = sinw / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / s - 1.0) + 2.0)
    sqA2a = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1) + (A - 1) * cosw + sqA2a)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
    b2 = A * ((A + 1) + (A - 1) * cosw - sqA2a)
    a0 = (A + 1) - (A - 1) * cosw + sqA2a
    a1 = 2 * ((A - 1) - (A + 1) * cosw)
    a2 = (A + 1) - (A - 1) * cosw - sqA2a
    return (np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))


def band_fraction(x: np.ndarray, sr: int, lo: float = 7000, hi: float = 8000) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    tot = spec.sum()
    return float(spec[(freqs >= lo) & (freqs < hi)].sum() / tot) if tot > 0 else 0.0


def main() -> None:
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {}
    done = skipped = 0
    for gen in SYNTH_GENERATORS:
        for p in sorted((ROOT / gen).rglob("*.wav")):
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if rel in log:
                skipped += 1
                continue
            x, sr = sf.read(str(p), dtype="float32")
            before = band_fraction(x, sr)
            b, a = high_shelf_coeffs(sr)
            y = lfilter(b, a, x).astype(np.float32)
            peak = float(np.max(np.abs(y)))
            if peak > 0.999:  # a cut cannot boost, but never write blind
                raise ValueError(f"unexpected peak {peak} after shelf on {p}")
            sf.write(str(p), y, sr, subtype="PCM_16")
            log[rel] = dict(op="high_shelf_cut", f0_hz=F0_HZ, gain_db=GAIN_DB,
                            shelf_s=SHELF_S, design="RBJ audio-EQ-cookbook biquad",
                            band_frac_7to8k_before=round(before, 8),
                            band_frac_7to8k_after=round(band_fraction(y, sr), 8))
            done += 1
    LOG.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"shelf applied to {done} clips, {skipped} already done -> {LOG}")

    print(f"\n7-8kHz energy fraction, per generator (after):")
    for gen in ("authentic",) + SYNTH_GENERATORS:
        vals = []
        for p in sorted((ROOT / gen).rglob("*.wav")):
            x, sr = sf.read(str(p), dtype="float32")
            vals.append(band_fraction(x, sr))
        v = np.array(vals)
        print(f"  {gen:10s} median={np.median(v):.2e}  min={v.min():.2e}  max={v.max():.2e}")


if __name__ == "__main__":
    main()
