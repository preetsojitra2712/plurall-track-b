#!/usr/bin/env python3
"""
Extract mean-pooled hidden states from a FROZEN wav2vec2 at several depths.

Why layer taps, not just the top: for anti-spoofing, the artifact signal
(vocoder phase texture, over-smooth spectral envelope) tends to live in low and
mid layers; the top layers are shaped by the ASR objective toward "what was
said", which is the one thing our content-matched design guarantees carries no
class signal. Which depth actually separates real from fake is an empirical
question -- this script produces the features, probe_layers.py answers it.

Indexing confirmed against the real model object (not assumed):
`output_hidden_states=True` yields 13 tensors for the 12-layer base model;
index 0 is the PRE-transformer feature-projection output and index k>=1 is
transformer layer k; hidden_states[-1] == last_hidden_state verified True.
Tap points: 0 (CNN/projection baseline) plus every 2nd transformer layer.

One clip at a time, deliberately no batching: clip lengths vary (2-13 s), and
batching would need padding plus an attention mask, and a mean-pool that
forgets the mask silently averages padding zeros into short clips' features --
a bug that produces plausible numbers. ~150 clips makes speed a non-issue.

    python scripts/extract_layer_features.py

Cache: data/features/w2v2_layers.npz (+ sidecar .json recording model name and
tap indices). Re-running skips clips already cached for the same model; if the
model name or tap set changes, the cache is rebuilt from scratch.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

MODEL_NAME = "facebook/wav2vec2-base-960h"
TAP_LAYERS = [0, 2, 4, 6, 8, 10, 12]  # 0 = pre-transformer projection output
EXPECTED_SR = 16000

MANIFEST = Path("data/manifest.csv")
CACHE = Path("data/features/w2v2_layers.npz")
META = Path("data/features/w2v2_layers.json")


def key_for(path: str, layer: int) -> str:
    # forward slashes so the key is stable across OSes
    return f"{Path(path).as_posix()}::L{layer}"


def load_cache() -> dict[str, np.ndarray]:
    if not (CACHE.exists() and META.exists()):
        return {}
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta.get("model") != MODEL_NAME or meta.get("tap_layers") != TAP_LAYERS:
        print(f"cache is for model={meta.get('model')} taps={meta.get('tap_layers')} "
              f"-- rebuilding from scratch")
        return {}
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} missing -- run make_manifest.py first")
    df = pd.read_csv(MANIFEST)

    cache = load_cache()
    todo = [p for p in df["path"]
            if not all(key_for(p, L) in cache for L in TAP_LAYERS)]
    print(f"{len(df)} clips in manifest; {len(df) - len(todo)} already cached; "
          f"{len(todo)} to extract")

    if todo:
        import torch
        from transformers import Wav2Vec2Model

        print(f"loading {MODEL_NAME} (frozen, eval mode) ...")
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)

        t0 = time.time()
        for n, path in enumerate(todo, 1):
            x, sr = sf.read(path, dtype="float32")
            if sr != EXPECTED_SR:
                raise ValueError(f"{path}: sr={sr}, expected {EXPECTED_SR}; "
                                 f"the tree is supposed to be uniform")
            if x.ndim != 1:
                raise ValueError(f"{path}: expected mono, got shape {x.shape}")
            # wav2vec2-base-960h was trained on zero-mean unit-var input
            # (do_normalize=True in its processor); replicate that exactly.
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.no_grad():
                out = model(torch.from_numpy(x)[None, :], output_hidden_states=True)
            for L in TAP_LAYERS:
                # mean over time -> (768,). No padding ever existed, so the
                # pool is exact by construction.
                v = out.hidden_states[L].squeeze(0).mean(dim=0).numpy()
                cache[key_for(path, L)] = v.astype(np.float32)
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)} clips  ({rate:.2f} clips/s, "
                      f"~{(len(todo) - n) / rate:.0f}s left)")
        wall = time.time() - t0

        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **cache)
        META.write_text(json.dumps(dict(model=MODEL_NAME, tap_layers=TAP_LAYERS,
                                        hidden_size=768, pooled="mean-over-time",
                                        input_norm="per-clip zero-mean unit-var"),
                                   indent=2), encoding="utf-8")
        print(f"\nwrote {CACHE} ({len(cache)} vectors) in {wall:.1f}s wall")
    else:
        wall = 0.0
        print("nothing to do -- cache is complete")

    per_split = df.groupby("split").size()
    print("\nclips per split (all cached):")
    for s, n in per_split.items():
        print(f"  {s:6s} {n}")
    print(f"tap layers: {TAP_LAYERS}  |  {len(cache)} vectors total  |  "
          f"extraction wall time {wall:.1f}s")


if __name__ == "__main__":
    main()
