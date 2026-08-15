#!/usr/bin/env python3
"""
SANITY GATE for the corrected LoRA run. Must pass BEFORE v2 is trusted.

Same control that caught the original bug: train ONLY a linear head (no LoRA)
on the cached, already-converged frozen layer-6 features -- where sklearn's
converged LogisticRegression scores val 1.000 / test 0.970 -- using the SAME
two-group optimizer recipe v2 uses (head at TrainConfig.head_lr, 72 steps,
cosine+warmup, accum 8, seed 0). Under the flat lr=1e-4 this scored val 0.546,
which is how the undertrained-head confound was proven.

If the corrected recipe does NOT get close to the sklearn ceiling here, the
optimizer is still too weak to train even a head, so nothing the end-to-end
LoRA run produces can be attributed to adaptation. Exit code 1 in that case so
the runner stops rather than proceeding to an uninterpretable experiment.

Gate: val_auc >= 0.95 and test_auc >= 0.90.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pk import finetune as FT  # noqa: E402
from pk import metrics as M  # noqa: E402

# 100 epochs, up from the original 8: the 8-epoch constraint was sized for CPU
# wall-clock, and the epoch-8 loss flattening was the cosine anneal ending, not
# convergence. Budget sweep (8/20/40/60/100/150) showed transfer gains collapse
# after ~100 (60->100 +0.006, 100->150 +0.004) while train loss on separable
# data has no finite asymptote to wait for. Same value in train_lora_probe_v2
# -- the gate must test the budget the real run uses.
EPOCHS, ACCUM, SEED, TAP = 100, 8, 0, 6
METRICS_JSON = Path("data/gpu_run_metrics.json")
GATE_VAL, GATE_TEST = 0.95, 0.90


def key_for(p, L=TAP):
    return f"{Path(p).as_posix()}::L{L}"


def main() -> None:
    df = pd.read_csv("data/manifest.csv")
    # manifest was built on Windows: backslash paths. On Linux a backslash is
    # not a separator, so normalise once here -- file IO and npz keys both
    # expect forward slashes (and Windows accepts them too).
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    with np.load("data/features/w2v2_layers.npz") as z:
        feats = {k: z[k] for k in z.files}

    def xy(split):
        sub = df[df.split == split]
        X = np.stack([feats[key_for(p)] for p in sub["path"]]).astype(np.float32)
        return torch.from_numpy(X), sub["label"].to_numpy()

    Xtr, ytr = xy("train")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    head = nn.Linear(768, 1)
    cfg = FT.TrainConfig()
    # v2's optimizer shape; with no LoRA present only the head group exists
    opt = torch.optim.Adam([dict(params=list(head.parameters()),
                                 lr=cfg.head_lr, base_lr=cfg.head_lr)])
    print(f"control: head-only, Adam lr={cfg.head_lr} (TrainConfig.head_lr), "
          f"{EPOCHS} epochs, accum={ACCUM}, seed={SEED}")
    lossf = nn.BCEWithLogitsLoss()

    steps_per_epoch = int(np.ceil(len(Xtr) / ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    step = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(Xtr))
        losses = []
        opt.zero_grad()
        for j, idx in enumerate(order, 1):
            logit = head(Xtr[int(idx)]).squeeze()
            loss = lossf(logit, torch.tensor(float(ytr[int(idx)])))
            (loss / ACCUM).backward()
            losses.append(float(loss.detach()))
            if j % ACCUM == 0 or j == len(order):
                mult = FT.cosine_schedule(step, total_steps, warmup_frac=0.1)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * mult
                opt.step()
                opt.zero_grad()
                step += 1
        print(f"  epoch {epoch}/{EPOCHS}  mean loss {np.mean(losses):.4f}  (chance=0.693)")

    out = {}
    for split in ("val", "test"):
        X, y = xy(split)
        with torch.no_grad():
            s = head(X).squeeze().numpy()
        out[split] = dict(auc=M.roc_auc(y, s), eer=M.eer(y, s)[0],
                          tpr01=M.tpr_at_fpr(y, s, target_fpr=0.1))
        print(f"{split:5s} auc={out[split]['auc']:.3f}  eer={out[split]['eer']:.3f}  "
              f"tpr@fpr=0.1={out[split]['tpr01']:.3f}")

    m = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    m["control_head_only"] = dict(val_auc=out["val"]["auc"], test_auc=out["test"]["auc"],
                                  test_eer=out["test"]["eer"],
                                  test_tpr01=out["test"]["tpr01"])
    METRICS_JSON.write_text(json.dumps(m, indent=2))

    print(f"\n(sklearn converged LR on identical features: val 1.000 / test 0.970)")
    if out["val"]["auc"] >= GATE_VAL and out["test"]["auc"] >= GATE_TEST:
        print(f"GATE PASS: control reaches the ceiling "
              f"(val {out['val']['auc']:.3f} >= {GATE_VAL}, "
              f"test {out['test']['auc']:.3f} >= {GATE_TEST}). "
              f"The optimizer recipe is strong enough to train a head; "
              f"the LoRA run is now interpretable.")
    else:
        print(f"GATE FAIL: val {out['val']['auc']:.3f} / test {out['test']['auc']:.3f} "
              f"still below ceiling ({GATE_VAL}/{GATE_TEST}). The optimizer recipe "
              f"remains too weak to train even a head -- STOPPING; do not interpret "
              f"any end-to-end LoRA result until this passes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
