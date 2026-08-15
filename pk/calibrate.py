"""
Calibration. Turns a raw model score into a probability you can print on an
evidence card and defend in a dispute.

Key operational point: a detector's reliability is CONDITIONAL on the channel the
media arrived through. A native camera upload and a WhatsApp forward are different
distributions, so fit calibrators PER SEGMENT (media type x quality bucket), not
globally. SegmentedCalibrator below does exactly that with a global fallback for
segments with too little data.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["logit", "sigmoid", "TemperatureScaler", "PlattScaler",
           "IsotonicCalibrator", "SegmentedCalibrator", "fit_calibrator",
           "GaussianScoreModel", "fit_gaussian_score_model"]

EPS = 1e-6


def logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)  # clamp: logit(0)=-inf poisons a sum
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                                   # numerically stable branch
    out[~pos] = ez / (1.0 + ez)
    return out


class TemperatureScaler:
    """Single-parameter scaling in logit space. Preserves ranking, so AUC is unchanged.

    This is the default. It cannot overfit, it needs very little held-out data,
    and it fixes the systematic overconfidence that logit-space fusion of
    correlated evidence dimensions produces.
    """

    def __init__(self):
        self.T = 1.0
        self.b = 0.0

    def fit(self, y_true, scores, grid=None):
        y = np.asarray(y_true, dtype=float)
        z = logit(scores)
        grid = grid if grid is not None else np.linspace(0.05, 10.0, 400)
        best, best_T, best_b = np.inf, 1.0, 0.0
        for T in grid:
            for b in (0.0,):
                p = sigmoid(z / T + b)
                nll = -np.mean(y * np.log(np.clip(p, EPS, 1)) +
                               (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
                if nll < best:
                    best, best_T, best_b = nll, T, b
        self.T, self.b = float(best_T), float(best_b)
        return self

    def transform(self, scores):
        return sigmoid(logit(scores) / self.T + self.b)

    __call__ = transform


class PlattScaler:
    """Logistic regression on the logit. Two parameters (slope + intercept)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, y_true, scores):
        self.lr.fit(logit(scores).reshape(-1, 1), np.asarray(y_true, dtype=int))
        return self

    def transform(self, scores):
        return self.lr.predict_proba(logit(scores).reshape(-1, 1))[:, 1]

    __call__ = transform


class IsotonicCalibrator:
    """Non-parametric, monotone. More flexible than Platt but needs more held-out
    data and can overfit small segments. Use when you have >~2k held-out samples."""

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_true, scores):
        self.ir.fit(np.asarray(scores, dtype=float), np.asarray(y_true, dtype=float))
        return self

    def transform(self, scores):
        return np.clip(self.ir.predict(np.asarray(scores, dtype=float)), EPS, 1 - EPS)

    __call__ = transform


def fit_calibrator(y_true, scores, kind: str = "temperature"):
    return {"temperature": TemperatureScaler,
            "platt": PlattScaler,
            "isotonic": IsotonicCalibrator}[kind]().fit(y_true, scores)


class GaussianScoreModel:
    """Generative two-Gaussian score model: calibrated posterior + open-set signal.

    Why this exists next to PlattScaler rather than instead of it: Platt fits one
    discriminative curve P(spoof|z) and can only ever say "which side". Fitting a
    Gaussian PER CLASS in logit space additionally says how TYPICAL the sample is
    of each class. A sample can sit confidently between the classes (low
    posterior information) or far outside BOTH (not credibly either class) --
    the second is the open-set case a discriminative calibrator is structurally
    blind to: an unseen synthesizer (ElevenLabs, Fish Audio, next month's model)
    does not owe our two training clusters membership.

    Posterior derivation, in one line from Bayes:
      P(spoof|z) = pi1*f1(z) / (pi1*f1(z) + pi0*f0(z))       divide by pi0*f0(z)
                 = sigmoid( log[f1(z)/f0(z)] + log[pi1/pi0] )
                 = sigmoid( LLR(z) + prior_log_odds )
    i.e. the logistic link is not a convenient squash, it IS the exact posterior
    for class-conditional densities; with equal priors, posterior = sigmoid(LLR).

    Note: with unequal variances the LLR is QUADRATIC in z, so the decision
    boundary can be two points, and the model can assign low spoof-posterior to
    samples far out on the spoof side's own tail. That is a feature of the
    generative view, not a bug -- extreme atypicality is exactly what the
    typicality output is for.
    """

    def __init__(self, mu0: float, s0: float, mu1: float, s1: float,
                 eps: float = 1e-12):
        self.mu0, self.s0 = float(mu0), float(s0)   # authentic (label 0)
        self.mu1, self.s1 = float(mu1), float(s1)   # spoof (label 1)
        self.eps = eps                               # wider than logit's default:
                                                     # converged probes emit
                                                     # probabilities within 1e-7 of
                                                     # 0/1, and clipping those at
                                                     # 1e-6 would pile a point mass
                                                     # at |z|=13.8 into the fit

    def _z(self, scores):
        return logit(scores, eps=self.eps)

    @staticmethod
    def _log_norm(z, mu, s):
        return -0.5 * np.log(2 * np.pi * s * s) - 0.5 * ((z - mu) / s) ** 2

    def llr(self, scores):
        """log f_spoof(z) - log f_auth(z). Positive = more spoof-like."""
        z = self._z(scores)
        return self._log_norm(z, self.mu1, self.s1) - self._log_norm(z, self.mu0, self.s0)

    def posterior(self, scores, prior_log_odds: float = 0.0):
        """Exact Bayes posterior (see class docstring). prior_log_odds defaults
        to 0 (equal priors) because the deployment base rate belongs to the
        operator, not to whatever class mix the training set happened to have."""
        return sigmoid(self.llr(scores) + prior_log_odds)

    def typicality(self, scores):
        """(sigmas_from_authentic, sigmas_from_spoof) per sample.

        min() of the pair large => the sample is not credibly from EITHER
        training distribution -- the open-set / novel-generator signal.
        """
        z = self._z(scores)
        return (np.abs(z - self.mu0) / self.s0,
                np.abs(z - self.mu1) / self.s1)

    def decision_boundaries(self):
        """z values where LLR(z)=0, i.e. posterior crosses 0.5 at equal priors.
        Quadratic in z (unequal variances), so 0, 1, or 2 real roots."""
        a = 0.5 / self.s0 ** 2 - 0.5 / self.s1 ** 2
        b = self.mu1 / self.s1 ** 2 - self.mu0 / self.s0 ** 2
        c = (0.5 * self.mu0 ** 2 / self.s0 ** 2
             - 0.5 * self.mu1 ** 2 / self.s1 ** 2
             + np.log(self.s0 / self.s1))
        if abs(a) < 1e-12:
            return [] if abs(b) < 1e-12 else [-c / b]
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        r = np.sqrt(disc)
        return sorted([(-b - r) / (2 * a), (-b + r) / (2 * a)])

    def bands(self, scores, posterior_thr: float = 0.5,
              abstain_sigma: float = 3.0, prior_log_odds: float = 0.0):
        """AUTHENTIC / SYNTHETIC / ABSTAIN labels.

        ABSTAIN fires when the sample is > abstain_sigma from BOTH fitted
        Gaussians (3-sigma default: outside what either training distribution
        credibly produces). It outranks the posterior: a confident-looking
        posterior between two densities the sample belongs to neither of is
        exactly the overconfidence this model exists to refuse.
        """
        t0, t1 = self.typicality(scores)
        post = self.posterior(scores, prior_log_odds)
        out = np.where(post >= posterior_thr, "SYNTHETIC", "AUTHENTIC").astype(object)
        out[np.minimum(t0, t1) > abstain_sigma] = "ABSTAIN"
        return out


def fit_gaussian_score_model(scores_authentic, scores_spoof,
                             eps: float = 1e-12) -> GaussianScoreModel:
    """Fit one 1-D Gaussian per class to LOGIT-transformed scores.

    Inputs are probabilities in (0,1) (e.g. predict_proba of a probe); the fit
    happens in logit space where scores are unbounded and closer to Gaussian.
    Fit on the TRAIN split only -- fitting on anything the model will be judged
    on is calibration leakage. Stds use ddof=1 and are floored to avoid a
    degenerate zero-width class.
    """
    z0 = logit(np.asarray(scores_authentic, dtype=float), eps=eps)
    z1 = logit(np.asarray(scores_spoof, dtype=float), eps=eps)
    if len(z0) < 3 or len(z1) < 3:
        raise ValueError("need at least 3 samples per class to fit a Gaussian")
    s0 = max(float(np.std(z0, ddof=1)), 1e-3)
    s1 = max(float(np.std(z1, ddof=1)), 1e-3)
    return GaussianScoreModel(float(np.mean(z0)), s0, float(np.mean(z1)), s1, eps=eps)


class SegmentedCalibrator:
    """Per-segment calibration with a global fallback.

    segments: array of hashable keys, e.g. ("video", "q_low"). Any segment with
    fewer than `min_n` samples falls back to the global calibrator rather than
    fitting a calibrator on noise.
    """

    def __init__(self, kind: str = "temperature", min_n: int = 200):
        self.kind, self.min_n = kind, min_n
        self.global_cal = None
        self.by_segment: dict = {}

    def fit(self, y_true, scores, segments):
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
        seg = np.asarray(segments, dtype=object)
        self.global_cal = fit_calibrator(y, s, self.kind)
        for key in np.unique(seg):
            m = seg == key
            if m.sum() >= self.min_n and len(np.unique(y[m])) == 2:
                self.by_segment[key] = fit_calibrator(y[m], s[m], self.kind)
        return self

    def transform(self, scores, segments):
        s = np.asarray(scores, dtype=float)
        seg = np.asarray(segments, dtype=object)
        out = self.global_cal.transform(s)
        for key, cal in self.by_segment.items():
            m = seg == key
            if m.any():
                out[m] = cal.transform(s[m])
        return out

    def coverage(self) -> dict:
        return {"segments_fitted": len(self.by_segment),
                "segments": sorted(map(str, self.by_segment.keys()))}
