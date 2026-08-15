#!/usr/bin/env python3
"""
Recover the reference transcript for every authentic clip already on disk.

Why: the generated positives have to be CONTENT-MATCHED to the bonafide set.
If the synthetic clips say different sentences than the real ones, a detector
can separate the classes on vocabulary, prosody-of-content, or utterance length
and never learn a synthesis artifact at all. Same sentences through both sides
is what forces the decision onto "was this generated".

Streams the same split as `fetch_librispeech_sample.py` but drops the audio
column entirely -- we already have the audio, we only need the text, and not
decoding is both faster and avoids the torchcodec dependency.

    python scripts/fetch_transcripts.py

Fails loudly if any clip on disk has no matching transcript in the stream,
rather than writing a partial map that silently shrinks the generated set.
"""
import json
import sys
from pathlib import Path

from datasets import Audio, load_dataset

REPO = "openslr/librispeech_asr"
CONFIG = "clean"
SPLIT = "validation"
MAX_SCAN = 6000

AUDIO_ROOT = Path("data/raw_audio/authentic")
OUT_PATH = Path("data/raw_audio/transcripts.json")


def main() -> None:
    clips = sorted(AUDIO_ROOT.rglob("*.wav"))
    if not clips:
        sys.exit(f"no clips under {AUDIO_ROOT}; run fetch_librispeech_sample.py first")
    wanted = {p.stem for p in clips}
    print(f"looking for transcripts for {len(wanted)} clips on disk")

    print(f"streaming {REPO} config={CONFIG!r} split={SPLIT!r} (text only) ...")
    ds = load_dataset(REPO, CONFIG, split=SPLIT, streaming=True)
    # Turn decoding OFF at the feature level before dropping the column.
    # remove_columns alone is not enough: the batch is formatted (and therefore
    # decoded) before the removal applies, so it still demands torchcodec.
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.remove_columns(["audio"])  # we already have the audio on disk

    found: dict[str, str] = {}
    scanned = 0
    for sample in ds:
        scanned += 1
        if scanned > MAX_SCAN:
            break
        uid = str(sample["id"])
        if uid in wanted and uid not in found:
            found[uid] = sample["text"]
            if len(found) == len(wanted):
                break

    missing = sorted(wanted - set(found))
    if missing:
        print(f"\n! {len(missing)} clip(s) had no transcript in the stream:")
        for m in missing:
            print(f"    {m}")
        sys.exit("refusing to write a partial transcript map")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(found, indent=2, sort_keys=True), encoding="utf-8")

    chars = [len(t) for t in found.values()]
    print(f"\nwrote {OUT_PATH}  ({len(found)} transcripts, scanned {scanned} items)")
    print(f"text length: min={min(chars)} max={max(chars)} mean={sum(chars) / len(chars):.0f} chars")
    sample_id = sorted(found)[0]
    print(f"example  {sample_id}: {found[sample_id][:90]!r}")


if __name__ == "__main__":
    main()
