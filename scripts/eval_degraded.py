#!/usr/bin/env python3
"""
Clean vs telephone-degraded, on the cross-synthesizer test split, for both
detectors we have: the frozen layer-6 sklearn probe and the LoRA-adapted
model from data/lora_v2_state.pt.

Both models were fit on CLEAN Kokoro+authentic train data and are evaluated
zero-shot on the degraded set -- no retraining. The question is purely "does
the score survive the channel", which is the deployment question: nobody
re-fits the detector per WhatsApp forward.

Outputs data/degraded_scores.csv (path,label,generator,degradation,score_*)
so the rows slot into pk/harness.py-style robustness reporting.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from pk import finetune as FT  # noqa: E402
from pk import metrics as M  # noqa: E402

MODEL_NAME = "facebook/wav2vec2-base-960h"
TAP = 6
CACHE = Path("data/features/w2v2_layers.npz")
STATE = Path("data/lora_v2_state.pt")
OUT_CSV = Path("data/degraded_scores.csv")


def key_for(p, L=TAP):
    return f"{Path(p).as_posix()}::L{L}"


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"{path}: sr={sr}")
    return (x - x.mean()) / (x.std() + 1e-7)


def extract(model, paths, dev, tag):
    feats = {}
    with torch.no_grad():
        for n, p in enumerate(paths, 1):
            x = torch.from_numpy(load_audio(p))[None, :].to(dev)
            out = model(x, output_hidden_states=True)
            feats[p] = out.hidden_states[TAP].squeeze(0).mean(dim=0).cpu().numpy()
            if n % 30 == 0:
                print(f"    {tag}: {n}/{len(paths)}")
    return feats


def report(name, y, s):
    auc = M.roc_auc(y, s)
    e = M.eer(y, s)[0]
    print(f"    {name:34s} auc={auc:.3f}  eer={e:.3f}")
    return auc, e


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfm = pd.read_csv("data/manifest.csv")
    dfm["path"] = dfm["path"].str.replace("\\", "/", regex=False)
    dd = pd.read_csv("data/manifest_degraded.csv")

    clean = dd[dd.degradation == "none"].reset_index(drop=True)
    deg = dd[dd.degradation == "telephone"].reset_index(drop=True)

    # ---------- frozen probe: sklearn on cached clean-train features ----------
    with np.load(CACHE) as z:
        cached = {k: z[k] for k in z.files}
    train = dfm[dfm.split == "train"]
    Xtr = np.stack([cached[key_for(p)] for p in train["path"]])
    ytr = train["label"].to_numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    probe.fit(Xtr, ytr)

    from transformers import Wav2Vec2Model
    print("extracting degraded-set features through FROZEN backbone ...")
    frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    frozen.eval()
    for p_ in frozen.parameters():
        p_.requires_grad_(False)
    frozen.to(dev)
    deg_frozen = extract(frozen, list(deg["path"]), dev, "frozen/degraded")

    y_clean = clean["label"].to_numpy()
    y_deg = deg["label"].to_numpy()
    Xc = np.stack([cached[key_for(p)] for p in clean["path"]])   # cached clean test
    Xd = np.stack([deg_frozen[p] for p in deg["path"]])

    print("\nFROZEN sklearn probe (fit on clean train):")
    fz_c = report("clean test", y_clean, probe.decision_function(Xc))
    fz_d = report("telephone-degraded test", y_deg, probe.decision_function(Xd))

    # ---------- LoRA-adapted model ----------
    print("\nrebuilding LoRA-adapted backbone from data/lora_v2_state.pt ...")
    state = torch.load(STATE, map_location="cpu")
    adapted = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    adapted.eval()
    for p_ in adapted.parameters():
        p_.requires_grad_(False)
    for i in range(6):
        FT.inject_lora(adapted.encoder.layers[i], target_substrings=("q_proj", "v_proj"),
                       r=state["meta"]["r"], alpha=state["meta"]["alpha"])
    own = dict(adapted.named_parameters())
    for n, t in state["lora"].items():
        with torch.no_grad():
            own[n].copy_(t)
    for p_ in adapted.parameters():
        p_.requires_grad_(False)
    adapted.to(dev)
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(state["head"])
    head.to(dev)

    ad_clean = extract(adapted, list(clean["path"]), dev, "adapted/clean")
    ad_deg = extract(adapted, list(deg["path"]), dev, "adapted/degraded")

    def head_scores(fmap, paths):
        X = torch.from_numpy(np.stack([fmap[p] for p in paths])).to(dev)
        with torch.no_grad():
            return head(X).squeeze(1).cpu().numpy()

    print("\nLoRA-ADAPTED model (backbone+head from lora_v2_state.pt):")
    lo_c = report("clean test", y_clean, head_scores(ad_clean, clean["path"]))
    lo_d = report("telephone-degraded test", y_deg, head_scores(ad_deg, deg["path"]))

    # ---------- side-by-side ----------
    print("\n" + "=" * 74)
    print("CLEAN vs TELEPHONE-DEGRADED -- test split (authentic + piper, unseen gen)")
    print("=" * 74)
    print(f"{'model':30s} {'clean auc':>9s} {'deg auc':>8s} {'d_auc':>7s} "
          f"{'clean eer':>9s} {'deg eer':>8s}")
    for nm, (ca, ce), (da, de) in [("frozen sklearn probe", fz_c, fz_d),
                                   ("LoRA-adapted (v2)", lo_c, lo_d)]:
        print(f"{nm:30s} {ca:9.3f} {da:8.3f} {da - ca:+7.3f} {ce:9.3f} {de:8.3f}")

    rows = []
    sc_c_fz = probe.decision_function(Xc)
    sc_d_fz = probe.decision_function(Xd)
    sc_c_lo = head_scores(ad_clean, clean["path"])
    sc_d_lo = head_scores(ad_deg, deg["path"])
    for (sub, s_fz, s_lo) in [(clean, sc_c_fz, sc_c_lo), (deg, sc_d_fz, sc_d_lo)]:
        for i in range(len(sub)):
            r = sub.iloc[i]
            rows.append(dict(path=r["path"], label=int(r["label"]),
                             generator=r["generator"], degradation=r["degradation"],
                             score_frozen=float(s_fz[i]), score_lora=float(s_lo[i])))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nscores with degradation column -> {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
