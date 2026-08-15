#!/usr/bin/env python3
"""
CORRECTED LoRA run: v1 with exactly one experimental change -- the optimizer.

v1's flaw, proven by control: one flat lr=1e-4 for BOTH the zero-init LoRA
adapters and the freshly-initialised head. 72 Adam steps at that LR cannot
train even the 769-param head (control on already-converged frozen features
scored val 0.546 vs sklearn's 1.000), so v1's collapse measured optimisation
budget, not adaptation. pk.finetune.TrainConfig documents the fix this module
always intended: head_lr = 1e-3, 10x the adapter LR, because a head starting
from random init must move faster than an adapter that starts as a no-op.

Changes vs train_lora_probe.py, and NOTHING else:
  1. two Adam param groups -- LoRA A/B at cfg.lr, head at cfg.head_lr, both
     read from an actual FT.TrainConfig instance, not restated as magic
     numbers (the cosine schedule scales each group's own base LR);
  2. model, head, and every tensor on the CUDA device (this is the GPU run;
     the script refuses to run without CUDA rather than silently crawling);
  3. adapter+head state saved at the end so probe_adapted_features.py can
     re-examine the adapted representation without retraining;
  4. metrics appended to data/gpu_run_metrics.json for the final table.
Unchanged by design: r=8, alpha=16, accum=8, seed=0, eval()-mode backbone
throughout, adapted layers 0..5 only, mean-pool on hidden_states[6].
Epochs raised 8 -> 100 under an explicit, recorded override: the original cap
was a CPU wall-clock constraint, and the gate proved 72 steps end mid-descent.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pk import finetune as FT  # noqa: E402
from pk import metrics as M  # noqa: E402

MODEL_NAME = "facebook/wav2vec2-base-960h"
MANIFEST = Path("data/manifest.csv")
RESULTS_CSV = Path("data/layer_probe_results.csv")
STATE_OUT = Path("data/lora_v2_state.pt")
METRICS_JSON = Path("data/gpu_run_metrics.json")
TAP = 6
ADAPT_LAYERS = range(6)
LORA_R, LORA_ALPHA = 8, 16
EPOCHS = 100  # raised from 8 (authorized override): the CPU-sized budget ended
              # before the descent did -- see control_head_only.py for the sweep
ACCUM = 8
SEED = 0


def append_metrics(name: str, row: dict) -> None:
    m = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    m[name] = row
    METRICS_JSON.write_text(json.dumps(m, indent=2))


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"{path}: sr={sr}")
    return (x - x.mean()) / (x.std() + 1e-7)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required -- this is the GPU run"
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(MANIFEST)
    # manifest was built on Windows: backslash paths. Normalise once -- on
    # Linux a backslash is not a separator, and forward slashes work on both.
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    splits = {s: df[df.split == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    for s, sub in splits.items():
        print(f"{s:6s} n={len(sub):3d}  neg={int((sub.label == 0).sum()):3d}  "
              f"pos={int((sub.label == 1).sum()):3d}  "
              f"pos from {sorted(sub.loc[sub.label == 1, 'generator'].unique()) or '-'}")

    print("\ncaching audio in RAM ...")
    audio = {p: load_audio(p) for p in df["path"]}

    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()  # permanent: dropout + randomly-init'd SpecAugment embed stay off
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for i in ADAPT_LAYERS:
        replaced += FT.inject_lora(model.encoder.layers[i],
                                   target_substrings=("q_proj", "v_proj"),
                                   r=LORA_R, alpha=LORA_ALPHA)
    assert replaced == len(list(ADAPT_LAYERS)) * 2
    bad = [n for n, p in model.named_parameters() if p.requires_grad
           and not (n.endswith(".A") or n.endswith(".B"))]
    assert not bad, f"non-LoRA trainables leaked: {bad}"

    head = nn.Linear(768, 1)
    model.to(dev)
    head.to(dev)

    print("\n" + FT.trainable_report(model))
    tr_b, tot_b = FT.count_params(model)
    tr_h = sum(p.numel() for p in head.parameters())
    print(f"with head: trainable {tr_b + tr_h:,} / {tot_b + tr_h:,} "
          f"({100 * (tr_b + tr_h) / (tot_b + tr_h):.4f}%)")

    # THE fix: two param groups at the documented TrainConfig rates.
    cfg = FT.TrainConfig()  # lr=1e-4 adapters, head_lr=1e-3 -- documented defaults
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam([
        dict(params=lora_params, lr=cfg.lr, base_lr=cfg.lr),
        dict(params=list(head.parameters()), lr=cfg.head_lr, base_lr=cfg.head_lr),
    ])
    print(f"\noptimizer: Adam, two groups -- LoRA lr={cfg.lr} (TrainConfig.lr), "
          f"head lr={cfg.head_lr} (TrainConfig.head_lr)")
    lossf = nn.BCEWithLogitsLoss()

    def pooled_layer6(x: np.ndarray) -> torch.Tensor:
        out = model(torch.from_numpy(x)[None, :].to(dev), output_hidden_states=True)
        return out.hidden_states[TAP].squeeze(0).mean(dim=0)

    train = splits["train"]
    steps_per_epoch = int(np.ceil(len(train) / ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    print(f"training: {EPOCHS} epochs x {len(train)} clips, accum={ACCUM} "
          f"-> {total_steps} optimiser steps\n")

    t0 = time.time()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train))
        losses = []
        opt.zero_grad()
        for j, idx in enumerate(order, 1):
            row = train.iloc[int(idx)]
            logit = head(pooled_layer6(audio[row["path"]])).squeeze()
            loss = lossf(logit, torch.tensor(float(row["label"]), device=dev))
            (loss / ACCUM).backward()
            losses.append(float(loss.detach()))
            if j % ACCUM == 0 or j == len(order):
                mult = FT.cosine_schedule(step, total_steps, warmup_frac=0.1)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * mult
                opt.step()
                opt.zero_grad()
                step += 1
        print(f"  epoch {epoch}/{EPOCHS}  mean train loss {np.mean(losses):.4f}  "
              f"(chance=0.693; {time.time() - t0:.0f}s elapsed)")
    train_wall = time.time() - t0
    print(f"\ntraining wall time: {train_wall:.1f}s")

    def scores_for(split: str):
        sub = splits[split]
        ss = np.zeros(len(sub))
        with torch.no_grad():
            for i in range(len(sub)):
                ss[i] = float(head(pooled_layer6(audio[sub.iloc[i]["path"]])))
        return sub["label"].to_numpy(), ss

    results = {}
    for split in ("val", "test"):
        y, s = scores_for(split)
        results[split] = dict(auc=M.roc_auc(y, s), eer=M.eer(y, s)[0],
                              tpr01=M.tpr_at_fpr(y, s, target_fpr=0.1))
        print(f"{split:5s} auc={results[split]['auc']:.3f}  "
              f"eer={results[split]['eer']:.3f}  "
              f"tpr@fpr=0.1={results[split]['tpr01']:.3f}")

    state = {"lora": {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
             "head": {k: v.cpu() for k, v in head.state_dict().items()},
             "meta": dict(model=MODEL_NAME, tap=TAP, r=LORA_R, alpha=LORA_ALPHA,
                          epochs=EPOCHS, seed=SEED,
                          lr=cfg.lr, head_lr=cfg.head_lr)}
    torch.save(state, STATE_OUT)
    print(f"\nadapter+head state -> {STATE_OUT}")

    append_metrics("lora_v2_end_to_end", dict(
        val_auc=results["val"]["auc"], test_auc=results["test"]["auc"],
        test_eer=results["test"]["eer"], test_tpr01=results["test"]["tpr01"],
        train_wall_s=round(train_wall, 1)))

    frozen = pd.read_csv(RESULTS_CSV).set_index("layer").loc[TAP]
    print(f"\n{'':26s} {'val_auc':>8s} {'test_auc':>9s} {'test_eer':>9s} {'tpr@0.1':>8s}")
    print(f"{'frozen probe (sklearn)':26s} {frozen['val_auc']:8.3f} "
          f"{frozen['test_auc']:9.3f} {frozen['test_eer']:9.3f} "
          f"{frozen['test_tpr@fpr=0.1']:8.3f}")
    print(f"{'LoRA v2 (two-group Adam)':26s} {results['val']['auc']:8.3f} "
          f"{results['test']['auc']:9.3f} {results['test']['eer']:9.3f} "
          f"{results['test']['tpr01']:8.3f}")


if __name__ == "__main__":
    main()
