"""
The evaluation harness.

This is the artefact I would build first at a deepfake company, before touching a
model. In a category where the adversary re-rolls every few weeks, you cannot
safely change anything without a number you trust, and every other decision
depends on that number.

What it produces:
  * a PER-GENERATOR matrix of TPR at fixed FPR (never one aggregate scalar --
    an aggregate hides the failure that actually kills you, which is total
    blindness to one popular new model)
  * a robustness CURVE (recall vs JPEG quality)
  * calibration (ECE / Brier) per segment
  * bootstrap confidence intervals
  * a RELEASE GATE: no regression on authentic media, mirroring Plurall's own
    changelog language for the v3.1 -> v3.2 promotion
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M

__all__ = ["EvalConfig", "EvalResult", "evaluate", "release_gate", "to_markdown"]


@dataclass
class EvalConfig:
    fprs: tuple = (0.001, 0.01, 0.05)
    primary_fpr: float = 0.01
    bootstrap_n: int = 500
    seed: int = 0
    min_rows_per_group: int = 30      # below this, report but do not gate on it


@dataclass
class EvalResult:
    overall: dict
    per_generator: list
    per_family: list
    per_quality: list
    robustness: list
    meta: dict

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def _group_metrics(df: pd.DataFrame, neg_scores: np.ndarray, cfg: EvalConfig, key: str) -> list:
    """Per-group TPR uses the GLOBAL negative pool to set the threshold.

    This matters and is easy to get wrong: a per-generator FPR computed against
    that generator's own negatives is meaningless, because a generator group has
    no negatives. The operating threshold is a property of the authentic
    distribution, so fix it globally, then measure each generator's recall at it.
    """
    rows = []
    thresholds = {f: M.threshold_at_fpr(
        np.zeros(len(neg_scores)), neg_scores, f) for f in cfg.fprs}
    for name, g in df[df.label == 1].groupby(key, dropna=False):
        row = {key: str(name), "n": int(len(g))}
        for f in cfg.fprs:
            row[f"tpr@fpr={f:g}"] = float((g["score"].values >= thresholds[f]).mean())
        pooled_y = np.concatenate([np.ones(len(g)), np.zeros(len(neg_scores))])
        pooled_s = np.concatenate([g["score"].values, neg_scores])
        row["auc"] = M.roc_auc(pooled_y, pooled_s)
        row["low_n"] = bool(len(g) < cfg.min_rows_per_group)
        rows.append(row)
    return sorted(rows, key=lambda r: r.get(f"tpr@fpr={cfg.primary_fpr:g}", 0))


def evaluate(df: pd.DataFrame, cfg: EvalConfig | None = None,
             prob_col: str | None = None) -> EvalResult:
    """df needs: label, score, generator, family, quality. Optional: degradation, prob.

    `score` may be an uncalibrated model output. `prob_col`, if given, must be
    calibrated -- that is the column ECE is computed on.
    """
    cfg = cfg or EvalConfig()
    required = {"label", "score", "generator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eval frame missing columns: {missing}")
    df = df.copy()
    for c in ("family", "quality", "degradation"):
        if c not in df.columns:
            df[c] = "unknown"

    y = df["label"].values.astype(int)
    s = df["score"].values.astype(float)
    neg = s[y == 0]

    overall = M.summary(y, s, df[prob_col].values if prob_col else None, cfg.fprs)
    pt, lo, hi = M.bootstrap_ci(y, s, M.roc_auc, n=cfg.bootstrap_n, seed=cfg.seed)
    overall["auc_ci95"] = [lo, hi]

    # robustness curve: performance as a function of degradation bucket
    robustness = []
    for name, g in df.groupby("degradation", dropna=False):
        gy, gs = g["label"].values.astype(int), g["score"].values.astype(float)
        if len(np.unique(gy)) < 2:
            continue
        robustness.append({
            "degradation": str(name), "n": int(len(g)),
            "auc": M.roc_auc(gy, gs),
            f"tpr@fpr={cfg.primary_fpr:g}": M.tpr_at_fpr(gy, gs, cfg.primary_fpr),
        })

    per_quality = []
    for name, g in df.groupby("quality", dropna=False):
        gy, gs = g["label"].values.astype(int), g["score"].values.astype(float)
        if len(np.unique(gy)) < 2:
            continue
        row = {"quality": str(name), "n": int(len(g)), "auc": M.roc_auc(gy, gs)}
        if prob_col:
            row["ece"] = M.ece(gy, g[prob_col].values)
        per_quality.append(row)

    return EvalResult(
        overall=overall,
        per_generator=_group_metrics(df, neg, cfg, "generator"),
        per_family=_group_metrics(df, neg, cfg, "family"),
        per_quality=per_quality,
        robustness=sorted(robustness, key=lambda r: -r["auc"]),
        meta={"ts": time.time(), "rows": int(len(df)), "config": asdict(cfg)},
    )


def release_gate(candidate: EvalResult, incumbent: EvalResult, cfg: EvalConfig | None = None,
                 max_authentic_regression: float = 0.0,
                 max_per_generator_drop: float = 0.03) -> dict:
    """Would you promote this model? Answer in code, not in a meeting.

    Two gates, both taken from how Plurall describe their own promotions:
      1. NO REGRESSION ON AUTHENTIC MEDIA -- the threshold required to hit the
         target FPR must not have to move upward, i.e. the model must not have
         become more trigger-happy on real content.
      2. No single generator family may drop more than max_per_generator_drop.
         An aggregate improvement that blinds you to one generator is a
         regression, not a win.
    """
    cfg = cfg or EvalConfig()
    k = f"tpr@fpr={cfg.primary_fpr:g}"
    reasons, ok = [], True

    d_overall = candidate.overall[k] - incumbent.overall[k]
    if d_overall < -1e-9:
        ok = False
        reasons.append(f"overall {k} regressed by {abs(d_overall):.4f}")

    inc = {r["generator"]: r for r in incumbent.per_generator}
    for r in candidate.per_generator:
        if r["low_n"] or r["generator"] not in inc:
            continue
        d = r[k] - inc[r["generator"]][k]
        if d < -max_per_generator_drop:
            ok = False
            reasons.append(f"generator '{r['generator']}' dropped {abs(d):.3f} on {k}")

    thr_key = f"thr@fpr={cfg.primary_fpr:g}"
    if thr_key in candidate.overall and thr_key in incumbent.overall:
        # informational: a big threshold shift means recalibration is mandatory
        reasons.append(f"note: operating threshold moved "
                       f"{incumbent.overall[thr_key]:.4f} -> {candidate.overall[thr_key]:.4f}")

    return {"promote": ok, "delta_overall": d_overall, "reasons": reasons}


def to_markdown(res: EvalResult, cfg: EvalConfig | None = None, title: str = "Eval report") -> str:
    cfg = cfg or EvalConfig()
    k = f"tpr@fpr={cfg.primary_fpr:g}"
    L = [f"# {title}", "",
         f"`n={res.meta['rows']}`  ·  "
         f"`pos={res.overall['n_pos']}`  ·  `neg={res.overall['n_neg']}`", "",
         "## Overall", "",
         "| metric | value |", "|---|---|"]
    for key, v in res.overall.items():
        if isinstance(v, (int, float)):
            L.append(f"| {key} | {v:.4f} |")
        elif isinstance(v, list):
            L.append(f"| {key} | [{v[0]:.4f}, {v[1]:.4f}] |")

    L += ["", "## Per generator (worst first -- read this column, not the average)", "",
          f"| generator | n | {k} | auc | |", "|---|---|---|---|---|"]
    for r in res.per_generator:
        flag = " ⚠ low-n" if r["low_n"] else (" 🔴 BLIND" if r["auc"] < 0.6 else
                                              (" ⚠ weak" if r["auc"] < 0.8 else ""))
        L.append(f"| {r['generator']} | {r['n']} | {r[k]:.3f} | {r['auc']:.3f} |{flag} |")

    if res.robustness:
        L += ["", "## Robustness by degradation", "",
              f"| degradation | n | auc | {k} |", "|---|---|---|---|"]
        for r in res.robustness:
            L.append(f"| {r['degradation']} | {r['n']} | {r['auc']:.3f} | {r[k]:.3f} |")

    if res.per_quality:
        L += ["", "## Calibration by quality bucket", "",
              "| quality | n | auc | ece |", "|---|---|---|---|"]
        for r in res.per_quality:
            L.append(f"| {r['quality']} | {r['n']} | {r['auc']:.3f} | "
                     f"{r.get('ece', float('nan')):.4f} |")

    blind = [r["generator"] for r in res.per_generator if r["auc"] < 0.6 and not r["low_n"]]
    weak = [r["generator"] for r in res.per_generator if 0.6 <= r["auc"] < 0.8 and not r["low_n"]]
    L += ["", "## Coverage gaps", "",
          f"- **BLIND** (AUC < 0.60, at or near chance): {blind if blind else 'none'}",
          f"- **weak** (AUC 0.60–0.80): {weak if weak else 'none'}", "",
          "Aggregate metrics hide these rows. Total blindness to one popular new",
          "generator is the failure mode that actually costs a customer money."]
    return "\n".join(L)
