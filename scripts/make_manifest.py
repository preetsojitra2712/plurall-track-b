#!/usr/bin/env python3
"""
Build a manifest from a directory tree, then split it without leakage.

Expected layout (either works):
    root/authentic/<source_id>/*.jpg
    root/<generator>/<source_id>/*.jpg

    python scripts/make_manifest.py --root /data/faces \
        --holdout sdxl newgen-x --out manifest.csv

If your data is flat, pass --source-from-filename and it will use the filename
stem up to the first underscore as source_id. Getting source_id right is the
whole game: it is what stops the same actor appearing on both sides of a split.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pk import data as D

FAMILY = {
    "stylegan": "gan", "stylegan2": "gan", "stylegan3": "gan", "progan": "gan",
    "sdxl": "diffusion", "sd15": "diffusion", "midjourney": "diffusion",
    "flux": "diffusion", "dalle": "diffusion", "dit": "diffusion",
    "faceswap": "faceswap", "deepfacelab": "faceswap", "simswap": "faceswap",
    "inswapper": "faceswap", "faceshifter": "faceswap",
    "reenact": "reenactment", "fomm": "reenactment", "heygen": "reenactment",
    "wav2lip": "lipsync", "authentic": "authentic",
    # Audio TTS. Kept as separate families by ARCHITECTURE, not lumped under a
    # single "tts": cross-synthesizer generalisation is the whole Track B
    # question, and a family label that cannot tell a flow-based VITS decoder
    # from a StyleTTS2/ISTFTNet one hides exactly the split you need to report.
    "kokoro": "tts_styletts2",   # StyleTTS2 + ISTFTNet decoder
    "piper": "tts_vits",         # VITS conditional VAE + normalising flows
}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".wav", ".flac", ".mp3"}
MEDIA = {".mp4": "video", ".mov": "video", ".wav": "audio", ".flac": "audio", ".mp3": "audio"}


def family_of(gen: str) -> str:
    g = gen.lower()
    for k, v in FAMILY.items():
        if k in g:
            return v
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--holdout", nargs="*", default=[], help="generators to hold out entirely")
    ap.add_argument("--source-from-filename", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        rel = p.relative_to(root).parts
        generator = rel[0] if rel else "unknown"
        if args.source_from_filename or len(rel) < 3:
            source_id = p.stem.split("_")[0]
        else:
            source_id = rel[1]
        rows.append(dict(
            path=str(p), label=0 if generator == "authentic" else 1,
            generator=generator, family=family_of(generator),
            source_id=source_id,
            media_type=MEDIA.get(p.suffix.lower(), "image"),
            quality="native",
        ))

    if not rows:
        sys.exit(f"no media found under {root}")

    df = pd.DataFrame(rows)
    df = D.group_split(df, D.SplitSpec(holdout_generators=args.holdout,
                                       val_fraction=args.val_fraction, seed=args.seed))
    rep = D.leakage_report(df, strict=True)

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df):,} rows)")
    print("\nsplit sizes:\n" + df.groupby("split").size().to_string())
    print("\nby family:\n" + D.balance_report(df).to_string())
    print("\nleakage:", rep)
    print("\ngenerators:", sorted(df.generator.unique()))
    if args.holdout:
        print(f"held out entirely (test only): {args.holdout}")

    # A leakage-clean split can still be useless. If a split holds only one
    # class you cannot compute TPR, EER or AUC on it at all -- and leakage_report
    # will happily call that "clean", because nothing leaked. Say so loudly.
    print("\nclass coverage per split:")
    single = []
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        if sub.empty:
            continue
        n0, n1 = int((sub.label == 0).sum()), int((sub.label == 1).sum())
        ok = n0 > 0 and n1 > 0
        print(f"  {split:6s} label0={n0:4d}  label1={n1:4d}   "
              f"{'OK  both classes' if ok else '*** SINGLE CLASS ***'}")
        if not ok:
            single.append(split)
    if single:
        print(f"\n!! {', '.join(single)} contain only ONE class. No threshold-based")
        print("!! metric (TPR@FPR, EER, AUC) is computable there. The split is")
        print("!! leakage-clean but NOT usable for evaluation.")
    else:
        print("\nevery split contains both classes -- metrics are computable.")


if __name__ == "__main__":
    main()
