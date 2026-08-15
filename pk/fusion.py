"""
Late fusion of the six evidence dimensions into one verdict.

Mirrors Plurall's published contract:
  dimensions : AI Model, Spectral, Diffusion, Temporal, EXIF, Web Intelligence
  card verdict enum : AUTHENTIC | SUSPICIOUS | SYNTHETIC | PLAUSIBLE | STRIPPED
  fused thresholds  : >= 0.85 SYNTHETIC, >= 0.5 SUSPICIOUS (tenant-overridable)
  confidence        : "the calibrated probability that the label is correct"

Design commitments, and why:
  * LATE, not early. Modalities are heterogeneous, missingness is the norm
    (STRIPPED EXIF, no temporal axis on a still), and evidence cards
    structurally require a per-dimension score.
  * Fuse in LOGIT space. Averaging probabilities is not combining evidence;
    log-odds add.
  * AVAILABILITY MASK, never zero-fill. Filling a missing card with 0.5 silently
    drags the verdict toward the abstention band.
  * Keep the head LINEAR as long as possible. w_d * logit(p_d) IS the number you
    print on the card, so interpretability here is a product spec, not a
    modelling preference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import logit, sigmoid

__all__ = ["DIMENSIONS", "Evidence", "FusionHead", "verdict_from_prob", "VERDICTS"]

DIMENSIONS = ("ai_model", "spectral", "diffusion", "temporal", "exif", "web_intel")
VERDICTS = ("AUTHENTIC", "SUSPICIOUS", "SYNTHETIC")


@dataclass
class Evidence:
    """One scan's worth of per-dimension output."""
    scores: dict           # dim -> calibrated probability in [0,1]
    available: dict        # dim -> bool
    context: dict = None   # media_type, quality bucket, codec, ... for the gate

    def vector(self):
        s = np.array([self.scores.get(d, 0.5) for d in DIMENSIONS], dtype=float)
        a = np.array([bool(self.available.get(d, False)) for d in DIMENSIONS], dtype=float)
        return s, a


def verdict_from_prob(p: float, syn: float = 0.85, sus: float = 0.5) -> str:
    if p >= syn:
        return "SYNTHETIC"
    if p >= sus:
        return "SUSPICIOUS"
    return "AUTHENTIC"


class FusionHead:
    """Missing-aware logistic fusion, trained with modality dropout.

    Modality dropout is the whole trick: randomly mask dimensions during training
    so the head learns to produce sensible output from ANY subset of cards. Without
    it, the head silently assumes all six are present and degrades badly the first
    time EXIF comes back STRIPPED in production.
    """

    def __init__(self, dims=DIMENSIONS, l2: float = 1.0, dropout_p: float = 0.3, seed: int = 0):
        self.dims = tuple(dims)
        self.w = np.zeros(len(self.dims))
        self.b = 0.0
        self.l2 = l2
        self.dropout_p = dropout_p
        self.rng = np.random.default_rng(seed)
        self.temperature = 1.0

    # ---------------------------------------------------------------- training
    def fit(self, S, A, y, epochs: int = 300, lr: float = 0.1, dropout_reps: int = 4):
        """S: (n, d) calibrated per-dimension probs. A: (n, d) availability 0/1. y: (n,)."""
        S = np.asarray(S, dtype=float)
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        L = logit(S)

        for _ in range(epochs):
            gw = np.zeros_like(self.w)
            gb = 0.0
            for _ in range(dropout_reps):
                mask = A.copy()
                if self.dropout_p > 0:
                    drop = self.rng.random(A.shape) < self.dropout_p
                    mask = mask * (~drop)
                    # never drop every card at once -- that row carries no signal
                    empty = mask.sum(axis=1) == 0
                    if empty.any():
                        mask[empty] = A[empty]
                z = (L * mask) @ self.w + self.b
                p = sigmoid(z)
                err = p - y
                gw += ((L * mask) * err[:, None]).sum(axis=0) / len(y)
                gb += err.mean()
            gw /= dropout_reps
            gb /= dropout_reps
            gw += self.l2 * self.w / len(y)
            self.w -= lr * gw
            self.b -= lr * gb
        return self

    def fit_temperature(self, S, A, y, grid=None):
        """Post-fusion temperature scaling. Necessary because the evidence
        dimensions are CORRELATED (Spectral and Diffusion agree on diffusion
        outputs), and logit-space summation implicitly assumes independence --
        so the fused score comes out systematically overconfident."""
        z = self._raw_logit(S, A)
        y = np.asarray(y, dtype=float).ravel()
        grid = grid if grid is not None else np.linspace(0.2, 6.0, 300)
        best, bestT = np.inf, 1.0
        for T in grid:
            p = np.clip(sigmoid(z / T), 1e-6, 1 - 1e-6)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best:
                best, bestT = nll, T
        self.temperature = float(bestT)
        return self

    # --------------------------------------------------------------- inference
    def _raw_logit(self, S, A):
        L = logit(np.asarray(S, dtype=float))
        A = np.asarray(A, dtype=float)
        return (L * A) @ self.w + self.b

    def predict_proba(self, S, A):
        return sigmoid(self._raw_logit(S, A) / self.temperature)

    def score_one(self, ev: Evidence, syn: float = 0.85, sus: float = 0.5) -> dict:
        """Full API-shaped result for a single scan, including per-card contributions."""
        s, a = ev.vector()
        p = float(self.predict_proba(s[None, :], a[None, :])[0])
        contrib = (logit(s) * a * self.w) / max(self.temperature, 1e-9)
        cards = []
        for i, d in enumerate(self.dims):
            cards.append({
                "dimension": d,
                "score": float(s[i]) if a[i] else None,
                "verdict": self._card_verdict(d, s[i], bool(a[i])),
                "contribution": float(contrib[i]),
            })
        cards.sort(key=lambda c: -abs(c["contribution"]))
        return {
            "confidence": p,
            "verdict": verdict_from_prob(p, syn, sus),
            "fusion": {"weights": dict(zip(self.dims, self.w.tolist())),
                       "bias": self.b, "temperature": self.temperature},
            "evidence": cards,
            "driver": cards[0]["dimension"] if cards else None,
        }

    @staticmethod
    def _card_verdict(dim: str, score: float, available: bool) -> str:
        if not available:
            return "STRIPPED" if dim == "exif" else "PLAUSIBLE"
        if score >= 0.85:
            return "SYNTHETIC"
        if score >= 0.5:
            return "SUSPICIOUS"
        return "AUTHENTIC"

    # --------------------------------------------------------------- reporting
    def abstention_rate(self, S, A, syn: float = 0.85, sus: float = 0.5) -> float:
        """Share of traffic landing in the SUSPICIOUS band.

        Monitor this. A rising abstention rate is the cheapest early signal that
        something new has entered the stream, and it needs no labels at all.
        """
        p = self.predict_proba(S, A)
        return float(((p >= sus) & (p < syn)).mean())

    def disagreement(self, S, A) -> np.ndarray:
        """Per-row spread across available cards.

        THE novel-generator alarm. A new generator typically defeats SOME signals
        and not others, so inter-dimension disagreement spikes BEFORE any accuracy
        metric moves -- and unlike accuracy, it requires no labels.
        """
        S = np.asarray(S, dtype=float)
        A = np.asarray(A, dtype=float)
        out = np.zeros(len(S))
        for i in range(len(S)):
            vals = S[i][A[i] > 0]
            out[i] = float(vals.std()) if len(vals) > 1 else 0.0
        return out
