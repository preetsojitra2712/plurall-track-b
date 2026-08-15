#!/usr/bin/env python3
"""
LoRA adaptation of wav2vec2 layers 1-6 + linear head on layer 6, vs the frozen
layer-6 probe.

The question this answers: the frozen probe said layer 6 carries signal that
partially transfers to an unseen synthesizer (test AUC 0.970). Does letting
low/mid layers ADAPT -- 0.16% of parameters via LoRA on q/v projections --
improve that cross-synthesizer number, or does adaptation just sharpen the
in-domain (Kokoro) fingerprint? Both outcomes are informative; the second is
the classic trap and finding it here would be a result, not a failure.

Module names were confirmed against the real model before injection:
encoder.layers[k].attention.{q_proj,v_proj}, Linear(768,768) -- matching
pk.finetune.inject_lora's default target_substrings ("qkv" simply never hits).
hidden_states[6] is the output of encoder.layers[5], so "adapt layers 1-6"
means injecting into encoder.layers[0..5] and layers 7-12 stay frozen; the
forward still runs through all 12 (per plan, simplicity over the ~2x forward
saving), but no gradient path exists above layer 6.

Design choices worth defending:
- Backbone stays in eval() mode THROUGHOUT training. Wav2Vec2 in train() mode
  activates encoder dropout (p=0.1) and SpecAugment time-masking -- and the
  mask embedding (masked_spec_embed) is RANDOMLY INITIALISED in this
  checkpoint (the load report says so), so train() would inject an untrained
  random vector into the very hidden states we probe. Gradients flow fine in
  eval mode; only the stochastic layers are disabled.
- One clip per forward, gradients accumulated to an effective batch of 8:
  identical reasoning to extract_layer_features.py -- variable-length clips
  plus padding plus an unmasked mean-pool is a silent bug, so padding is
  never created.
- Mean-pool over time on hidden_states[6], exactly mirroring the frozen
  extraction, so the comparison isolates ADAPTATION as the only change.
- Cosine LR schedule with warmup (pk.finetune.cosine_schedule): LoRA B=0 at
  init makes the adapter a no-op at step 0, and warmup is what lets a
  zero-initialised adapter leave the frozen point smoothly.

    python scripts/train_lora_probe.py
"""
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
TAP = 6                 # hidden_states index; = output of encoder.layers[5]
ADAPT_LAYERS = range(6)  # encoder.layers[0..5]
LORA_R, LORA_ALPHA = 8, 16
EPOCHS = 8
LR = 1e-4
ACCUM = 8               # effective batch size, via accumulation not padding
SEED = 0


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"{path}: sr={sr}")
    # same per-clip normalisation as extract_layer_features.py
    return (x - x.mean()) / (x.std() + 1e-7)


def pooled_layer6(model, x: np.ndarray) -> torch.Tensor:
    out = model(torch.from_numpy(x)[None, :], output_hidden_states=True)
    return out.hidden_states[TAP].squeeze(0).mean(dim=0)  # (768,)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(MANIFEST)
    splits = {s: df[df.split == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    for s, sub in splits.items():
        print(f"{s:6s} n={len(sub):3d}  neg={int((sub.label == 0).sum()):3d}  "
              f"pos={int((sub.label == 1).sum()):3d}  "
              f"pos from {sorted(sub.loc[sub.label == 1, 'generator'].unique()) or '-'}")

    print("\ncaching audio in RAM (150 clips, ~40 MB) ...")
    audio = {p: load_audio(p) for p in df["path"]}

    from transformers import Wav2Vec2Model
    print(f"loading {MODEL_NAME} ...")
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()  # PERMANENT: see docstring (dropout + random SpecAugment embed)

    # freeze everything first -- the same blanket loop freeze_all_but_layernorm
    # opens with (we do NOT unfreeze LayerNorms: this is a pure-LoRA run)
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for i in ADAPT_LAYERS:
        replaced += FT.inject_lora(model.encoder.layers[i],
                                   target_substrings=("q_proj", "v_proj"),
                                   r=LORA_R, alpha=LORA_ALPHA)
    print(f"\ninjected LoRA (r={LORA_R}, alpha={LORA_ALPHA}, dropout=0) into "
          f"{replaced} Linear modules across encoder.layers[0..5]")
    assert replaced == len(list(ADAPT_LAYERS)) * 2, "expected q+v per layer"

    # every trainable backbone tensor must be a LoRA A/B and live in layers 0-5
    bad = [n for n, p in model.named_parameters() if p.requires_grad
           and not (n.endswith(".A") or n.endswith(".B"))]
    assert not bad, f"non-LoRA trainables leaked: {bad}"
    bad_depth = [n for n, p in model.named_parameters() if p.requires_grad
                 and int(n.split("layers.")[1].split(".")[0]) > 5]
    assert not bad_depth, f"LoRA above layer 6: {bad_depth}"

    head = nn.Linear(768, 1)  # separate module, so inject_lora cannot freeze it

    print("\n" + FT.trainable_report(model))
    tr_b, tot_b = FT.count_params(model)
    tr_h = sum(p.numel() for p in head.parameters())
    total = tot_b + tr_h
    print(f"\nwith head: trainable {tr_b + tr_h:,} / {total:,} "
          f"({100 * (tr_b + tr_h) / total:.4f}%)  "
          f"[backbone LoRA {tr_b:,} + head {tr_h:,}]")

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    train = splits["train"]
    steps_per_epoch = int(np.ceil(len(train) / ACCUM))
    total_steps = steps_per_epoch * EPOCHS

    print(f"\ntraining: {EPOCHS} epochs x {len(train)} clips, accum={ACCUM} "
          f"-> {total_steps} optimiser steps, Adam lr={LR} cosine+warmup\n")
    t0 = time.time()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train))
        losses = []
        opt.zero_grad()
        for j, idx in enumerate(order, 1):
            row = train.iloc[int(idx)]
            feat = pooled_layer6(model, audio[row["path"]])
            logit = head(feat).squeeze()
            loss = lossf(logit, torch.tensor(float(row["label"])))
            (loss / ACCUM).backward()
            losses.append(float(loss))
            if j % ACCUM == 0 or j == len(order):
                for g in opt.param_groups:
                    g["lr"] = LR * FT.cosine_schedule(step, total_steps, warmup_frac=0.1)
                opt.step()
                opt.zero_grad()
                step += 1
        print(f"  epoch {epoch}/{EPOCHS}  mean train loss {np.mean(losses):.4f}  "
              f"({time.time() - t0:.0f}s elapsed)")
    train_wall = time.time() - t0

    print(f"\ntraining wall time: {train_wall:.1f}s ({train_wall / 60:.1f} min)")

    # ---- evaluation, same metric calls as probe_layers.py
    def scores_for(split: str) -> tuple[np.ndarray, np.ndarray]:
        sub = splits[split]
        ss = np.zeros(len(sub))
        with torch.no_grad():
            for i in range(len(sub)):
                ss[i] = float(head(pooled_layer6(model, audio[sub.iloc[i]["path"]])))
        return sub["label"].to_numpy(), ss

    rows = {}
    for split in ("val", "test"):
        y, s = scores_for(split)
        rows[split] = dict(auc=M.roc_auc(y, s), eer=M.eer(y, s)[0],
                           tpr01=M.tpr_at_fpr(y, s, target_fpr=0.1))
        print(f"{split:5s} auc={rows[split]['auc']:.3f}  eer={rows[split]['eer']:.3f}  "
              f"tpr@fpr=0.1={rows[split]['tpr01']:.3f}")

    # ---- like-for-like comparison against the frozen layer-6 probe
    frozen = pd.read_csv(RESULTS_CSV).set_index("layer").loc[TAP]
    print("\n" + "=" * 76)
    print(f"COMPARISON -- layer {TAP} tap, frozen linear probe vs LoRA-adapted (1-6)")
    print("=" * 76)
    print(f"{'':24s} {'val_auc':>8s} {'test_auc':>9s} {'test_eer':>9s} {'test_tpr@0.1':>13s}")
    print(f"{'frozen probe':24s} {frozen['val_auc']:8.3f} {frozen['test_auc']:9.3f} "
          f"{frozen['test_eer']:9.3f} {frozen['test_tpr@fpr=0.1']:13.3f}")
    print(f"{'LoRA r=8 + head':24s} {rows['val']['auc']:8.3f} {rows['test']['auc']:9.3f} "
          f"{rows['test']['eer']:9.3f} {rows['test']['tpr01']:13.3f}")
    d_auc = rows["test"]["auc"] - frozen["test_auc"]
    print(f"\ndelta test AUC: {d_auc:+.3f}")
    if abs(d_auc) < 0.02:
        print("VERDICT: adaptation did NOT meaningfully move cross-synthesizer AUC.")
        print("A null result on 150 clips is expected and honest -- the frozen probe")
        print("was already near ceiling in-domain, so the only room to 'improve' was")
        print("fitting Kokoro harder, which does not transfer by construction.")
    elif d_auc > 0:
        print(f"VERDICT: adaptation improved cross-synthesizer AUC by {d_auc:+.3f}.")
        print("Check the val gap before celebrating: if val also rose to ceiling,")
        print("part of this could still be in-domain sharpening.")
    else:
        print(f"VERDICT: adaptation HURT cross-synthesizer AUC by {d_auc:+.3f} --")
        print("the classic trap: in-domain gain, out-of-domain loss. This is the")
        print("result the rehearsal/PEFT literature predicts for naive adaptation.")


if __name__ == "__main__":
    main()
