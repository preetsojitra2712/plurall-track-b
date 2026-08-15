#!/usr/bin/env python3
"""
The optimizer-confound-free answer to "what did LoRA do to the representation".

Loads the v2-adapted backbone (base + saved LoRA state), freezes everything,
re-extracts mean-pooled layer-6 hidden states for all 150 clips, and fits a
fresh sklearn LogisticRegression to convergence -- EXACTLY the recipe that
produced the original frozen-probe baseline (val 1.000 / test 0.970). Any
difference between that baseline and this run is attributable to what LoRA
changed in the representation, because the readout is identical and converged
in both cases. This is the row of the final table that answers the real
question; the end-to-end numbers also reflect optimizer quality.

Also prints the final four-row comparison table from data/gpu_run_metrics.json.
"""
import json
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
STATE = Path("data/lora_v2_state.pt")
METRICS_JSON = Path("data/gpu_run_metrics.json")
RESULTS_CSV = Path("data/layer_probe_results.csv")
TAP = 6
ADAPT_LAYERS = range(6)


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"{path}: sr={sr}")
    return (x - x.mean()) / (x.std() + 1e-7)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("data/manifest.csv")
    # manifest was built on Windows: backslash paths. Normalise once -- on
    # Linux a backslash is not a separator, and forward slashes work on both.
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    state = torch.load(STATE, map_location="cpu")
    print(f"loaded {STATE}  (meta: {state['meta']})")

    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for i in ADAPT_LAYERS:
        FT.inject_lora(model.encoder.layers[i], target_substrings=("q_proj", "v_proj"),
                       r=state["meta"]["r"], alpha=state["meta"]["alpha"])

    own = dict(model.named_parameters())
    lora_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(state["lora"]) == lora_names, "saved LoRA keys do not match injection"
    with torch.no_grad():
        for n, t in state["lora"].items():
            own[n].copy_(t)
    for p in model.parameters():          # frozen from here on -- pure extraction
        p.requires_grad = False
    model.to(dev)
    print(f"restored {len(state['lora'])} LoRA tensors; backbone frozen on {dev}")

    feats = {}
    with torch.no_grad():
        for n, path in enumerate(df["path"], 1):
            x = torch.from_numpy(load_audio(path))[None, :].to(dev)
            out = model(x, output_hidden_states=True)
            feats[path] = out.hidden_states[TAP].squeeze(0).mean(dim=0).cpu().numpy()
            if n % 50 == 0:
                print(f"  extracted {n}/{len(df)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def xy(split):
        sub = df[df.split == split]
        return np.stack([feats[p] for p in sub["path"]]), sub["label"].to_numpy()

    Xtr, ytr = xy("train")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(Xtr, ytr)

    out = {}
    for split in ("val", "test"):
        X, y = xy(split)
        s = clf.decision_function(X)
        out[split] = dict(auc=M.roc_auc(y, s), eer=M.eer(y, s)[0],
                          tpr01=M.tpr_at_fpr(y, s, target_fpr=0.1))
        print(f"{split:5s} auc={out[split]['auc']:.3f}  eer={out[split]['eer']:.3f}  "
              f"tpr@fpr=0.1={out[split]['tpr01']:.3f}")

    m = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    m["sklearn_on_adapted"] = dict(val_auc=out["val"]["auc"], test_auc=out["test"]["auc"],
                                   test_eer=out["test"]["eer"],
                                   test_tpr01=out["test"]["tpr01"])
    METRICS_JSON.write_text(json.dumps(m, indent=2))

    # ---------------- final table ----------------
    frozen = pd.read_csv(RESULTS_CSV).set_index("layer").loc[TAP]
    rows = [
        ("frozen probe (sklearn, no LoRA)",
         frozen["val_auc"], frozen["test_auc"], frozen["test_eer"],
         frozen["test_tpr@fpr=0.1"]),
        ("control: head-only Adam (gate)",
         m["control_head_only"]["val_auc"], m["control_head_only"]["test_auc"],
         m["control_head_only"]["test_eer"], m["control_head_only"]["test_tpr01"]),
        ("LoRA v2 end-to-end Adam",
         m["lora_v2_end_to_end"]["val_auc"], m["lora_v2_end_to_end"]["test_auc"],
         m["lora_v2_end_to_end"]["test_eer"], m["lora_v2_end_to_end"]["test_tpr01"]),
        ("sklearn on LoRA-adapted feats",
         out["val"]["auc"], out["test"]["auc"], out["test"]["eer"],
         out["test"]["tpr01"]),
    ]
    print("\n" + "=" * 78)
    print("FINAL TABLE -- layer-6 tap, cross-synthesizer test = piper (never trained on)")
    print("=" * 78)
    print(f"{'':34s} {'val_auc':>8s} {'test_auc':>9s} {'test_eer':>9s} {'tpr@0.1':>8s}")
    for name, va, ta, te, tp in rows:
        print(f"{name:34s} {va:8.3f} {ta:9.3f} {te:9.3f} {tp:8.3f}")

    d = out["test"]["auc"] - frozen["test_auc"]
    print(f"\nRow 4 vs row 1 is the real question -- identical converged readout,")
    print(f"only the representation differs. delta test AUC: {d:+.3f}")
    if abs(d) < 0.02:
        print("ANSWER: adaptation neither helped nor hurt cross-synthesizer transfer")
        print("materially at this scale.")
    elif d > 0:
        print("ANSWER: adaptation IMPROVED the representation for cross-synthesizer")
        print("transfer.")
    else:
        print("ANSWER: adaptation DEGRADED the representation for cross-synthesizer")
        print("transfer -- in-domain fitting bought at the held-out generator's cost.")


if __name__ == "__main__":
    main()
