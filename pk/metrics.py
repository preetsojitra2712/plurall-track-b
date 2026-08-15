"""
Detection metrics that matter for a deepfake/fraud detector.

Design principle: accuracy is not in this file on purpose. With a rare positive
class, accuracy is a number that hides the failure. The headline metric is
TPR at a fixed, low FPR -- the operating point the customer actually feels.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "roc_auc", "tpr_at_fpr", "threshold_at_fpr", "partial_auc",
    "eer", "ece", "brier", "precision_recall_f1", "bootstrap_ci",
    "summary",
]


def _as_arrays(y_true, y_score):
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN -- fix upstream, do not impute silently")
    return y, s


def roc_auc(y_true, y_score) -> float:
    """AUC via the Mann-Whitney U statistic. Handles ties with average ranks.

    Interpretation: P(random positive scores above random negative).
    """
    y, s = _as_arrays(y_true, y_score)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_at_fpr(y_true, y_score, target_fpr: float) -> float:
    """Smallest threshold whose realised FPR on the negatives is <= target_fpr.

    This is the (1 - target_fpr) quantile of the NEGATIVE score distribution.
    Returns +inf if no threshold achieves it (i.e. target_fpr < 1/n_neg).
    """
    y, s = _as_arrays(y_true, y_score)
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return float("nan")
    k = int(np.floor(target_fpr * len(neg)))  # how many negatives we may flag
    if k <= 0:
        # cannot flag any negative -> threshold must exceed the max negative
        return float(np.nextafter(neg[-1], np.inf))
    return float(neg[len(neg) - k])


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.01) -> float:
    """THE headline metric. Recall achievable while flagging <= target_fpr of authentic media."""
    y, s = _as_arrays(y_true, y_score)
    thr = threshold_at_fpr(y, s, target_fpr)
    pos = s[y == 1]
    if len(pos) == 0 or not np.isfinite(thr):
        return 0.0 if len(pos) else float("nan")
    return float((pos >= thr).mean())


def partial_auc(y_true, y_score, max_fpr: float = 0.1, standardize: bool = True) -> float:
    """AUC restricted to FPR in [0, max_fpr]. Optionally McClish-standardised to [0.5, 1].

    Full AUC weights operating points nobody will ever run at. This does not.
    """
    y, s = _as_arrays(y_true, y_score)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / max(tps[-1], 1)])
    fpr = np.concatenate([[0.0], fps / max(fps[-1], 1)])
    mask = fpr <= max_fpr
    x, ytp = fpr[mask], tpr[mask]
    if len(x) < 2 or x[-1] < max_fpr:  # interpolate to the exact boundary
        idx = np.searchsorted(fpr, max_fpr)
        if idx < len(fpr):
            lo = max(idx - 1, 0)
            span = fpr[idx] - fpr[lo]
            frac = 0.0 if span == 0 else (max_fpr - fpr[lo]) / span
            x = np.concatenate([x, [max_fpr]])
            ytp = np.concatenate([ytp, [tpr[lo] + frac * (tpr[idx] - tpr[lo])]])
    area = float(np.trapezoid(ytp, x))
    if not standardize:
        return area
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    return float(0.5 * (1 + (area - min_area) / (max_area - min_area)))


def eer(y_true, y_score) -> tuple[float, float]:
    """Equal Error Rate and the threshold that achieves it.

    EER is the point where the false-acceptance rate equals the false-rejection
    rate. It is the ASVspoof convention and the vocabulary the audio side of this
    field expects -- using it signals you have read the literature.
    """
    y, s = _as_arrays(y_true, y_score)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    candidates = np.unique(s)
    best_gap, best_eer, best_thr = np.inf, 1.0, float("nan")
    for t in candidates:
        far = float(len(neg) - np.searchsorted(neg, t, side="left")) / n_neg
        frr = float(np.searchsorted(pos, t, side="left")) / n_pos
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer, best_thr = gap, (far + frr) / 2.0, float(t)
    return float(best_eer), best_thr


def ece(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins).

    Plurall define `confidence` as "the calibrated probability that the label is
    correct". That makes calibration a shipped product surface, so ECE is a
    release metric, not a diagnostic.
    """
    y, p = _as_arrays(y_true, y_prob)
    if p.min() < 0 or p.max() > 1:
        raise ValueError("ece() expects probabilities in [0, 1] -- calibrate before calling")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y_true, y_prob) -> float:
    y, p = _as_arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def precision_recall_f1(y_true, y_score, thr: float) -> dict:
    y, s = _as_arrays(y_true, y_score)
    pred = s >= thr
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fdr": fp / (tp + fp) if tp + fp else 0.0}


def bootstrap_ci(y_true, y_score, fn=roc_auc, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. Use this instead of reporting a bare point estimate.

    For A/B of two models on the SAME data, bootstrap the DIFFERENCE (paired) --
    it has far more power than comparing two independent intervals.
    """
    y, s = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        stats.append(fn(y[b], s[b]))
    if not stats:
        return float("nan"), float("nan"), float("nan")
    stats = np.array(stats, dtype=float)
    return float(fn(y, s)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_delta(y_true, score_a, score_b, fn=roc_auc, n: int = 1000,
                           alpha: float = 0.05, seed: int = 0):
    """CI on fn(B) - fn(A) with the SAME resampled rows for both. Use for model promotion."""
    y = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        r = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[r])) < 2:
            continue
        deltas.append(fn(y[r], b[r]) - fn(y[r], a[r]))
    d = np.array(deltas, dtype=float)
    point = fn(y, b) - fn(y, a)
    return float(point), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def summary(y_true, y_score, y_prob=None, fprs=(0.001, 0.01, 0.05)) -> dict:
    """One call -> the full operating picture. y_prob only if scores are calibrated."""
    out = {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(np.asarray(y_true) == 1)),
        "n_neg": int(np.sum(np.asarray(y_true) == 0)),
        "auc": roc_auc(y_true, y_score),
        "pauc@10%fpr": partial_auc(y_true, y_score, 0.10),
    }
    for f in fprs:
        out[f"tpr@fpr={f:g}"] = tpr_at_fpr(y_true, y_score, f)
        out[f"thr@fpr={f:g}"] = threshold_at_fpr(y_true, y_score, f)
    e, t = eer(y_true, y_score)
    out["eer"], out["eer_thr"] = e, t
    if y_prob is not None:
        out["ece"] = ece(y_true, y_prob)
        out["brier"] = brier(y_true, y_prob)
    return out
