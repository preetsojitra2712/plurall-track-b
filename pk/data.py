"""
Dataset construction. This is where most deepfake projects lose, silently.

Three rules encoded here:
  1. Split by GENERATOR and by SOURCE IDENTITY, never at random. A random split
     measures memorisation, because the same actor / the same source video / the
     same generator appears on both sides.
  2. Prefer PAIRED real-fake from the same source. WACV 2026 is explicit that
     this is what prevents shortcut learning -- otherwise the model learns
     "this lighting / this codec / this actor => fake".
  3. Assert no leakage after every split, in code, and fail loudly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["MANIFEST_COLUMNS", "validate_manifest", "group_split",
           "leakage_report", "phash64", "hamming64", "dedup_by_phash",
           "balance_report", "SplitSpec"]

# One row per media asset. Everything downstream reads this and nothing else.
MANIFEST_COLUMNS = {
    "path": str,          # location on disk
    "label": int,         # 1 = synthetic/manipulated, 0 = authentic
    "generator": str,     # e.g. "stylegan2", "sdxl", "faceswap-df", "authentic"
    "family": str,        # coarse: gan | diffusion | faceswap | reenactment | tts | vc | authentic
    "source_id": str,     # identity / source video -- THE grouping key for leakage
    "media_type": str,    # image | video | audio
    "quality": str,       # native | q_high | q_mid | q_low  (compression bucket)
    "split": str,         # train | val | test  (filled by group_split)
}


@dataclass
class SplitSpec:
    """Which generators are held out. Held-out generators are the ONLY honest
    estimate of what happens when a new model drops next Tuesday."""
    holdout_generators: list[str] = field(default_factory=list)
    val_fraction: float = 0.15
    seed: int = 0


def validate_manifest(df: pd.DataFrame, require_split: bool = False) -> pd.DataFrame:
    missing = [c for c in MANIFEST_COLUMNS if c != "split" and c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if require_split and "split" not in df.columns:
        raise ValueError("manifest has no 'split' column -- run group_split first")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0/1")
    if df["path"].duplicated().any():
        dupes = df.loc[df["path"].duplicated(), "path"].head(5).tolist()
        raise ValueError(f"duplicate paths in manifest, e.g. {dupes}")
    # authentic rows should be labelled 0 and vice versa -- catches label-flip bugs early
    bad = df[(df["generator"] == "authentic") & (df["label"] != 0)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows marked generator=authentic but label=1")
    return df


def group_split(df: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Assign train/val/test with generator-level AND source-level isolation.

    test  = every row whose generator is in holdout_generators, plus authentic
            rows whose source_id hashes into the test bucket
    val   = a source-disjoint slice of the remainder
    train = the rest
    """
    df = validate_manifest(df).copy()
    rng = np.random.default_rng(spec.seed)

    is_holdout_gen = df["generator"].isin(spec.holdout_generators)

    # Authentic media has no generator, so split it by source_id hash. Using a hash
    # rather than a shuffle makes the assignment stable across runs and machines.
    def bucket(sid: str) -> float:
        h = hashlib.blake2b(str(sid).encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / 2 ** 64

    frac = df["source_id"].map(bucket)
    authentic_test = (df["label"] == 0) & (frac < 0.2)

    df["split"] = "train"
    df.loc[is_holdout_gen | authentic_test, "split"] = "test"

    # val: source-disjoint from train, carved out of what's left
    remaining = df["split"] == "train"
    sources = np.array(df.loc[remaining, "source_id"].unique(), dtype=object)
    rng.shuffle(sources)
    n_val = max(1, int(len(sources) * spec.val_fraction))
    val_sources = set(sources[:n_val])
    df.loc[remaining & df["source_id"].isin(val_sources), "split"] = "val"

    return df


def leakage_report(df: pd.DataFrame, strict: bool = True) -> dict:
    """Assert the split is honest. Call this in CI, not just once by hand."""
    validate_manifest(df, require_split=True)
    out = {}
    splits = {s: g for s, g in df.groupby("split")}

    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        if a not in splits or b not in splits:
            continue
        shared_src = set(splits[a]["source_id"]) & set(splits[b]["source_id"])
        shared_gen = (set(splits[a].loc[splits[a].label == 1, "generator"]) &
                      set(splits[b].loc[splits[b].label == 1, "generator"]))
        out[f"{a}|{b}:shared_source_ids"] = len(shared_src)
        out[f"{a}|{b}:shared_generators"] = sorted(shared_gen)

    problems = [k for k, v in out.items()
                if k.endswith("shared_source_ids") and v > 0]
    problems += [k for k, v in out.items()
                 if k.endswith("shared_generators") and v and "train|test" in k]
    out["clean"] = not problems
    if strict and problems:
        raise AssertionError(f"SPLIT LEAKAGE: {problems}\n{out}")
    return out


def balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split, per-family counts. Look at this before you look at any metric."""
    return (df.groupby(["split", "family", "label"], dropna=False)
              .size().rename("n").reset_index()
              .pivot_table(index=["split", "family"], columns="label",
                           values="n", fill_value=0))


# ---------------------------------------------------------------- perceptual hash

def phash64(img_gray_32x32: np.ndarray) -> int:
    """64-bit DCT perceptual hash from a 32x32 grayscale array.

    Used for (a) de-duplicating the training set -- near-duplicates across a split
    boundary are leakage -- and (b) the Web Intelligence style near-duplicate
    lookup. The FTC's TAKE IT DOWN guidance specifically recommends hashing to
    stop removed content from reappearing, so this is a compliance primitive too.
    """
    from scipy.fftpack import dct
    a = np.asarray(img_gray_32x32, dtype=float)
    if a.shape != (32, 32):
        raise ValueError("expected a 32x32 grayscale array")
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()[1:]                      # drop DC: it only encodes brightness
    bits = flat > np.median(flat)
    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= (1 << i)
    return h


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_by_phash(hashes: dict, max_dist: int = 6, bands: int = 8, bits: int = 64):
    """Banded LSH candidate generation + exact verification.

    bands > max_dist gives a pigeonhole guarantee (any pair within max_dist must
    share at least one identical band). Fewer bands trades recall for speed.
    """
    band_bits = bits // bands
    buckets: dict = {}
    for hid, h in hashes.items():
        for b in range(bands):
            key = (b, (h >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(hid)
    seen, pairs = set(), []
    for ids in buckets.values():
        if len(ids) > 500:      # a degenerate bucket (e.g. all-black frames)
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = tuple(sorted((ids[i], ids[j])))
                if pair in seen:
                    continue
                seen.add(pair)
                d = hamming64(hashes[pair[0]], hashes[pair[1]])
                if d <= max_dist:
                    pairs.append((pair, d))
    return sorted(pairs, key=lambda x: x[1])
