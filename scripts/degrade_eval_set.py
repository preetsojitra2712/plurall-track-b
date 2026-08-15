#!/usr/bin/env python3
"""
Build the telephone-channel copy of the TEST split (authentic + piper).

Writes to data/raw_audio_degraded/ with the identical generator/source_id/
utterance_id layout; the clean files are never touched, because the deliverable
is a clean-vs-degraded comparison, not a replacement. Emits:

  data/raw_audio_degraded/degrade_log.json   per-clip recipe (reproducibility)
  data/manifest_degraded.csv                 clean + degraded test rows with a
                                             `degradation` column, fitting
                                             pk/harness.py robustness reporting

Only the test split is degraded: the question being answered is "does the
cross-synthesizer evaluation survive the deployment channel", not "train on
degraded audio" -- that would be a training intervention, out of scope here.
The channel is applied with one seeded engine over clips sorted by path, so
the whole set is reproducible end to end.
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pk import audio_degrade as AD  # noqa: E402

MANIFEST = Path("data/manifest.csv")
OUT_ROOT = Path("data/raw_audio_degraded")
LOG = OUT_ROOT / "degrade_log.json"
OUT_MANIFEST = Path("data/manifest_degraded.csv")
SEED = 0


def main() -> None:
    df = pd.read_csv(MANIFEST)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    test = df[df.split == "test"].sort_values("path").reset_index(drop=True)
    print(f"test split: {len(test)} clips "
          f"({int((test.label == 0).sum())} authentic, "
          f"{int((test.label == 1).sum())} spoof)")

    chan = AD.TelephoneChannel(seed=SEED)
    log = {}
    wrote = skipped = 0
    for _, row in test.iterrows():
        src = Path(row["path"])
        rel = src.relative_to("data/raw_audio")
        dest = OUT_ROOT / rel
        if dest.exists():
            skipped += 1
            continue
        x, sr = sf.read(str(src), dtype="float32")
        y, recipe = chan(x, sr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), y, sr, subtype="PCM_16")
        log[str(rel).replace("\\", "/")] = recipe
        wrote += 1
    if wrote:
        LOG.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"degraded {wrote} clips, {skipped} already present -> {OUT_ROOT}")
    print(f"recipes -> {LOG}")

    clean = test.copy()
    clean["degradation"] = "none"
    deg = test.copy()
    deg["degradation"] = "telephone"
    deg["path"] = deg["path"].str.replace("data/raw_audio/", "data/raw_audio_degraded/",
                                          regex=False)
    out = pd.concat([clean, deg], ignore_index=True)
    out.to_csv(OUT_MANIFEST, index=False)
    print(f"manifest with degradation column -> {OUT_MANIFEST} ({len(out)} rows)")
    print(out.groupby(["degradation", "label"]).size().to_string())


if __name__ == "__main__":
    main()
