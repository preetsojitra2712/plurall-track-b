#!/usr/bin/env python3
"""
30-second end-to-end smoke test on synthetic data.

Run this the moment you sit down at the interview desk. It proves your
environment works, and it gives you something concrete on screen to talk about
before you have written a line of their code.

    python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pk import calibrate as C
from pk import data as D
from pk import fusion as FU
from pk import harness as H
from pk import metrics as M

rng = np.random.default_rng(0)
line = lambda t: print(f"\n{'─' * 66}\n{t}\n{'─' * 66}")

# ---------------------------------------------------------------- 1. manifest
line("1  MANIFEST + LEAKAGE-SAFE SPLIT")
rows = []
gens = {"authentic": ("authentic", 0), "stylegan2": ("gan", 1), "sdxl": ("diffusion", 1),
        "faceswap-df": ("faceswap", 1), "newgen-x": ("diffusion", 1)}
for i in range(1000):
    g = list(gens)[i % 5]
    fam, lab = gens[g]
    rows.append(dict(path=f"/data/{i}.jpg", label=lab, generator=g, family=fam,
                     source_id=f"id{i % 80}", media_type="image",
                     quality=rng.choice(["q_high", "q_mid", "q_low"])))
man = D.group_split(pd.DataFrame(rows), D.SplitSpec(holdout_generators=["newgen-x"], seed=1))
print(man.groupby(["split"]).size().to_string())
print("\nleakage check:", D.leakage_report(man, strict=True))
print("→ 'newgen-x' is held out entirely. Splitting by generator and source_id is")
print("  the difference between measuring generalisation and measuring memorisation.")

# ------------------------------------------------------- 2. simulate a detector
line("2  SCORE + EVALUATE (per-generator, not aggregate)")
# Score the whole manifest so the per-generator table shows both the generators
# the model saw and the one it did not. In a real run this is the test split only.
sep = {"authentic": 0.0, "stylegan2": 2.3, "sdxl": 2.0, "faceswap-df": 1.7, "newgen-x": 0.1}
test = man.copy()
test["degradation"] = rng.choice(["none", "light", "heavy"], size=len(test), p=[.5, .3, .2])
raw = np.array([C.sigmoid(sep[g] + rng.normal(0, 1.0)) for g in test.generator])
raw = np.where(test.degradation.values == "heavy",
               np.clip(raw - 0.20 * test.label.values, 1e-4, 1 - 1e-4), raw)
test["score"] = raw
test["prob"] = raw                                    # pretend these are calibrated

cfg = H.EvalConfig(bootstrap_n=200)
res = H.evaluate(test, cfg, prob_col="prob")
print(H.to_markdown(res, cfg, title="Detector v0"))
print("→ Aggregate AUC looks respectable. The per-generator table is where the")
print("  actual failure lives: the held-out generator is at chance. That row is")
print("  the number to put on a dashboard, not the average.")

# ------------------------------------------------------------- 3. calibration
line("3  CALIBRATION (their `confidence` field is a calibrated probability)")
over = np.clip(C.sigmoid(2.5 * C.logit(test.score.values)), 1e-6, 1 - 1e-6)  # overconfident
cal = C.TemperatureScaler().fit(test.label.values, over)
fixed = cal.transform(over)
print(f"  ECE before : {M.ece(test.label.values, over):.4f}")
print(f"  ECE after  : {M.ece(test.label.values, fixed):.4f}   (T={cal.T:.2f})")
print(f"  AUC before : {M.roc_auc(test.label.values, over):.4f}")
print(f"  AUC after  : {M.roc_auc(test.label.values, fixed):.4f}  ← unchanged, by construction")
print("→ Temperature scaling is monotone, so it fixes calibration without touching ranking.")

# ------------------------------------------------------------------ 4. fusion
line("4  SIX-CARD LATE FUSION (missing-modality aware)")
n = 4000
truth = rng.integers(0, 2, n)
S = np.zeros((n, 6)); A = np.ones((n, 6))
for j, k in enumerate([1.6, 1.0, 1.2, 0.8, 0.4, 0.7]):
    S[:, j] = np.clip(C.sigmoid(k * (truth * 2 - 1) + rng.normal(0, 1.0, n)), 1e-4, 1 - 1e-4)
A[rng.random(n) < 0.35, 4] = 0.0                     # EXIF STRIPPED on 35% of traffic

head = FU.FusionHead(seed=0).fit(S, A, truth, epochs=400, lr=0.3).fit_temperature(S, A, truth)
p = head.predict_proba(S, A)
for j, d in enumerate(FU.DIMENSIONS):
    print(f"  {d:12s} solo AUC {M.roc_auc(truth, S[:, j]):.4f}   weight {head.w[j]:+.3f}")
print(f"  {'FUSED':12s}      AUC {M.roc_auc(truth, p):.4f}")
print(f"  abstention rate (0.5–0.85 band): {head.abstention_rate(S, A):.1%}")

ev = FU.Evidence(
    scores=dict(ai_model=.93, spectral=.88, diffusion=.91, temporal=.60, exif=.50, web_intel=.70),
    available=dict(ai_model=True, spectral=True, diffusion=True,
                   temporal=True, exif=False, web_intel=True))
out = head.score_one(ev)
print(f"\n  verdict     : {out['verdict']}  (confidence {out['confidence']:.3f})")
print(f"  driver card : {out['driver']}")
for c in out["evidence"]:
    sc = "  —  " if c["score"] is None else f"{c['score']:.2f}"
    print(f"    {c['dimension']:12s} {sc}  {c['verdict']:<10s} contrib {c['contribution']:+.3f}")

# -------------------------------------------------------------- 5. release gate
line("5  RELEASE GATE (no regression on any generator)")
bad = test.copy()
m = bad.generator == "faceswap-df"
bad.loc[m, "score"] = np.clip(bad.loc[m, "score"] - 0.35, 1e-4, 1 - 1e-4)
gate = H.release_gate(H.evaluate(bad, cfg), res, cfg)
print(f"  promote: {gate['promote']}")
for r in gate["reasons"]:
    print(f"    - {r}")
print("→ An aggregate improvement that blinds you to one generator is a regression.")

print("\n" + "=" * 66)
print("  SMOKE TEST COMPLETE — environment is working.")
print("=" * 66)
