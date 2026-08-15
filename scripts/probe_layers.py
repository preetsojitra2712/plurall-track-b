#!/usr/bin/env python3
"""
Layer-wise linear probe over frozen wav2vec2 features: WHERE does the
real-vs-synthetic signal live?

Method: for each tap layer independently, fit a logistic regression on TRAIN
(authentic + kokoro only), then score VAL (in-domain: same generator family as
train, disjoint sources) and TEST (cross-synthesizer: piper/VITS, an engine
never seen in training). The val curve says "is there a linearly readable
signal at this depth"; the test curve says "does that signal transfer to an
unseen synthesizer" -- and the GAP between them is the honest headline. A layer
that only wins in-domain has learned Kokoro's fingerprint, not synthesis.

The probe is deliberately linear (the fusion-head philosophy in CLAUDE.md): if
a linear readout on frozen features already separates the classes, that is
evidence about the representation, not about a classifier's capacity.

    python scripts/probe_layers.py

Outputs: data/layer_probe_results.csv, data/layer_probe_curve.png.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pk import metrics as M  # noqa: E402

CACHE = Path("data/features/w2v2_layers.npz")
META = Path("data/features/w2v2_layers.json")
MANIFEST = Path("data/manifest.csv")
OUT_CSV = Path("data/layer_probe_results.csv")
OUT_PNG = Path("data/layer_probe_curve.png")

# With 10 negatives in val and 10 in test, the realisable FPR grid is steps of
# 0.1 -- an "FPR=0.01" operating point does not exist on 10 negatives, and
# quoting tpr@1% would be a number with no support behind it. 10% is the
# tightest honest budget here, so that is what we report.
TARGET_FPR = 0.1
MIN_PER_CLASS = 5


def key_for(path: str, layer: int) -> str:
    return f"{Path(path).as_posix()}::L{layer}"


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    taps: list[int] = meta["tap_layers"]
    df = pd.read_csv(MANIFEST)
    with np.load(CACHE) as z:
        feats = {k: z[k] for k in z.files}

    missing = [p for p in df["path"] if key_for(p, taps[0]) not in feats]
    if missing:
        sys.exit(f"{len(missing)} clips missing from cache -- run "
                 f"extract_layer_features.py first (e.g. {missing[0]})")

    # ---- class counts first; a probe on a single-class split is meaningless
    print("class counts per split (before anything else):")
    fragile = False
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        n0, n1 = int((sub.label == 0).sum()), int((sub.label == 1).sum())
        gens = sorted(sub.loc[sub.label == 1, "generator"].unique())
        print(f"  {s:6s} neg={n0:3d}  pos={n1:3d}   positives from {gens or '-'}")
        if min(n0, n1) < MIN_PER_CLASS:
            fragile = True
            print(f"  !! {s} has fewer than {MIN_PER_CLASS} of one class -- every"
                  f" number on this split is fragile; treat it as indicative only")
    print(f"\nmetrics: roc_auc, eer, tpr@fpr={TARGET_FPR}.")
    print(f"tpr@fpr=0.01 is deliberately NOT reported: with 10 negatives per eval")
    print(f"split the realisable FPR grid moves in steps of 0.1, so a 1% operating")
    print(f"point does not exist on this data and quoting it would be fiction.\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def xy(split: str, layer: int):
        sub = df[df.split == split]
        X = np.stack([feats[key_for(p, layer)] for p in sub["path"]])
        return X, sub["label"].to_numpy()

    rows = []
    for L in taps:
        Xtr, ytr = xy("train", L)
        # Standardise on TRAIN stats only; 768-dim features on 67 rows need the
        # scaler and a bounded C to keep the probe honest.
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0))
        clf.fit(Xtr, ytr)

        row = dict(layer=L)
        for split in ("val", "test"):
            X, y = xy(split, L)
            s = clf.decision_function(X)
            row[f"{split}_auc"] = M.roc_auc(y, s)
            row[f"{split}_eer"] = M.eer(y, s)[0]  # (rate, thr) tuple
            if split == "test":
                row["test_tpr@fpr=0.1"] = M.tpr_at_fpr(y, s, target_fpr=TARGET_FPR)
        rows.append(row)
        print(f"  layer {L:2d}: val_auc={row['val_auc']:.3f}  "
              f"test_auc={row['test_auc']:.3f}  gap={row['val_auc'] - row['test_auc']:+.3f}")

    res = pd.DataFrame(rows)[["layer", "val_auc", "val_eer",
                              "test_auc", "test_eer", "test_tpr@fpr=0.1"]]
    res.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res.layer, res.val_auc, "o-", label="val AUC (in-domain: kokoro)")
    ax.plot(res.layer, res.test_auc, "s-",
            label="test AUC (cross-synthesizer: piper, unseen)")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("wav2vec2 hidden_states index (0 = pre-transformer projection)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Linear probe by depth, frozen wav2vec2-base-960h")
    ax.set_xticks(res.layer)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")

    # ---- the two verdicts, kept separate on purpose
    best_test = res.loc[res.test_auc.idxmax()]
    res["gap"] = (res.val_auc - res.test_auc).abs()
    best_gap = res.loc[res.gap.idxmin()]
    print(f"\nbest TEST AUC          : layer {int(best_test.layer)} "
          f"(test_auc={best_test.test_auc:.3f})")
    print(f"smallest |val-test| gap: layer {int(best_gap.layer)} "
          f"(val={best_gap.val_auc:.3f}, test={best_gap.test_auc:.3f}, "
          f"gap={best_gap.gap:.3f})")
    print("\nThese answer different questions. Best-on-test can be luck on one")
    print("small held-out set; the smallest-gap layer is the one whose in-domain")
    print("performance you can most nearly trust as a forecast of transfer.")
    if fragile:
        print("\nREMINDER: a split above is under the minimum class count; do not")
        print("promote anything on these numbers alone.")


if __name__ == "__main__":
    main()
