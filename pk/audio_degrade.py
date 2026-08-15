"""
Telephone-channel degradation for audio -- the audio twin of pk/degrade.py.

The failure this prevents: evaluating an anti-spoofing model only on clean
studio-adjacent audio. The deployment channel for voice fraud is the phone
network: 300-3400 Hz bandwidth, G.711 companding, jitter-buffer dropouts, and
a room between the mouth and the microphone. A detector whose artifact cues
live above 4 kHz or in fine temporal structure can be excellent on clean data
and blind after one codec pass -- the audio version of the Deepfake-Eval-2024
AUC collapse. Same non-negotiable as the image engine: the IDENTICAL parameter
distribution is applied regardless of class, and every clip's recipe is logged.

Honesty notes, also recorded in data/SOURCES.md:
- codec: ffmpeg is not on this machine, so G.711 mu-law is implemented
  directly in numpy. That is NOT an approximation -- G.711 is exactly 8 kHz +
  mu-law companding + 8-bit quantisation, all of which happen here. AMR-NB and
  Opus round-trips would need ffmpeg and are simply not available, not faked.
- RIR: no measured impulse-response set is installed, so the RIR is SYNTHETIC
  (exponentially-decaying filtered noise), a standard cheap approximation. It
  produces plausible reverberant smearing but is not a measured room.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

__all__ = ["mu_law_roundtrip", "codec_roundtrip", "bandpass_telephone",
           "packet_loss", "synthetic_rir", "convolve_rir", "TelephoneChannel"]

G711_SR = 8000
MU = 255.0


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g)


def mu_law_roundtrip(x: np.ndarray, sr: int) -> np.ndarray:
    """G.711 mu-law codec round-trip, implemented literally.

    Resample to 8 kHz, compand with mu=255, quantise to 8 bits, expand,
    resample back. This is the actual G.711 algorithm, not a stand-in: the
    codec IS companding + quantisation at 8 kHz. Output length matches input.
    """
    n = len(x)
    y = _resample(x, sr, G711_SR)
    y = np.clip(y, -1.0, 1.0)
    comp = np.sign(y) * np.log1p(MU * np.abs(y)) / np.log1p(MU)
    q = np.round((comp + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0   # 8-bit grid
    exp = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(MU))) / MU
    out = _resample(exp, G711_SR, sr)
    # resample_poly can return n+-1 samples; pin the length
    if len(out) >= n:
        return np.asarray(out[:n], dtype=np.float32)
    return np.asarray(np.pad(out, (0, n - len(out))), dtype=np.float32)


def codec_roundtrip(x: np.ndarray, sr: int, codec: str = "g711_mulaw") -> np.ndarray:
    """Codec dispatch. Only G.711 mu-law is available on this machine (no
    ffmpeg), and it is implemented for real in numpy. Asking for a codec that
    would need ffmpeg raises rather than silently substituting."""
    if codec == "g711_mulaw":
        return mu_law_roundtrip(x, sr)
    raise ValueError(f"codec {codec!r} needs ffmpeg, which is not available; "
                     f"only 'g711_mulaw' is implemented honestly here")


def bandpass_telephone(x: np.ndarray, sr: int,
                       lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    """300-3400 Hz Butterworth bandpass -- the classic PSTN passband.

    Single-pass (not zero-phase): a real channel has phase distortion, and
    filtfilt's zero-phase would be cleaner than reality.
    """
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return np.asarray(sosfilt(sos, x), dtype=np.float32)


def packet_loss(x: np.ndarray, sr: int, loss_rate: float = 0.05,
                burst_ms: float = 20.0, rng: np.random.Generator | None = None
                ) -> np.ndarray:
    """Zero out random bursts, VoIP-jitter style.

    Concealment choice, documented: dropped frames are SILENCE (zeros), the
    harshest standard option. Real endpoints run packet-loss concealment that
    interpolates the gap; silence is the worst case and makes the degradation
    unambiguous rather than codec-specific. burst_ms=20 matches a typical
    RTP packet payload.
    """
    rng = rng or np.random.default_rng()
    y = x.copy()
    frame = max(1, int(sr * burst_ms / 1000.0))
    n_frames = len(y) // frame
    if n_frames == 0:
        return y
    drop = rng.random(n_frames) < loss_rate
    for i in np.nonzero(drop)[0]:
        y[i * frame:(i + 1) * frame] = 0.0
    return y


def synthetic_rir(sr: int, rt60: float = 0.3,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """SYNTHETIC room impulse response -- disclosed approximation, not a room.

    Exponentially-decaying white noise with a unit direct path: the textbook
    cheap stand-in when no measured RIR set is available. Captures the
    reverberant tail's statistical effect (temporal smearing, comb-like
    colouration) but none of a real room's geometry, early reflections, or
    directivity. rt60 is the -60 dB decay time.
    """
    rng = rng or np.random.default_rng()
    n = max(8, int(sr * rt60))
    t = np.arange(n) / sr
    env = np.exp(-6.908 * t / rt60)            # -60 dB at t = rt60
    tail = rng.standard_normal(n) * env
    rir = np.zeros(n, dtype=np.float32)
    rir[0] = 1.0                               # direct path
    rir[1:] = 0.35 * tail[1:] / (np.max(np.abs(tail[1:])) + 1e-9)
    return rir


def convolve_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve, trim back to the dry length, restore the dry RMS level.

    Level restoration matters: reverb adds energy, and after the tree-wide
    loudness equalisation a louder-because-reverberant clip would reintroduce
    level as a class cue the moment only one side gets the room.
    """
    dry_rms = float(np.sqrt(np.mean(x ** 2)))
    y = np.convolve(x, rir)[: len(x)]
    wet_rms = float(np.sqrt(np.mean(y ** 2)))
    if wet_rms > 0 and dry_rms > 0:
        y = y * (dry_rms / wet_rms)
    return np.asarray(y, dtype=np.float32)


class TelephoneChannel:
    """Compound telephone-channel simulator, mirroring DegradationEngine.

        chan = TelephoneChannel(seed=0)
        y, recipe = chan(x, sr)

    Ops are applied in physical order -- room, then channel bandpass, then
    codec, then network loss -- each with a sampled probability/parameters.
    The SAME distribution applies to every clip regardless of class (the
    shortcut rule), and `recipe` is returned for per-clip logging so any
    degraded sample can be reproduced exactly.
    """

    def __init__(self, seed: int = 0, p_rir: float = 0.5, p_codec: float = 0.9,
                 p_loss: float = 0.7):
        self.rng = np.random.default_rng(seed)
        self.p_rir, self.p_codec, self.p_loss = p_rir, p_codec, p_loss

    def __call__(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        r = self.rng
        y = np.asarray(x, dtype=np.float32)
        recipe = []
        if r.random() < self.p_rir:
            rt60 = float(r.uniform(0.15, 0.6))
            y = convolve_rir(y, synthetic_rir(sr, rt60, r))
            recipe.append(f"rir(rt60={rt60:.2f},synthetic)")
        y = bandpass_telephone(y, sr)
        recipe.append("bandpass(300-3400)")
        if r.random() < self.p_codec:
            y = mu_law_roundtrip(y, sr)
            recipe.append("g711_mulaw")
        if r.random() < self.p_loss:
            rate = float(r.uniform(0.02, 0.10))
            burst = float(r.choice([20.0, 40.0]))
            y = packet_loss(y, sr, rate, burst, r)
            recipe.append(f"ploss(rate={rate:.3f},burst={burst:.0f}ms)")
        return y, "|".join(recipe)
