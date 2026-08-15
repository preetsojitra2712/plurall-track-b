#!/usr/bin/env python3
"""
Synthesize the generated-positive side of the Track B set: the same 50
LibriSpeech sentences, spread across FOUR voices per engine, through two
architecturally distinct TTS engines.

Content matching is the point. If the synthetic clips said different sentences
than the bonafide ones, a detector could separate the classes on vocabulary or
utterance length and never learn a synthesis artifact. Same text through both
engines forces the decision onto "was this generated", and using two engines
stops the model learning one vendor's fingerprint and calling it "synthetic".

FOUR voices per engine, not one. A single voice per generator collapses
source_id to a single value, and group_split then has to move that generator
between splits as one indivisible block -- which is exactly how the first run
put all 100 positives in train and left test and val with no positives at all.
Identity diversity inside a generator is what makes a usable split possible.

    python scripts/generate_tts_positives.py

Idempotent: an (engine, voice, utterance) triple whose wav already exists is
skipped. Provenance is rebuilt from the plan each run so it can never describe
voices that are no longer on disk.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TRANSCRIPTS = Path("data/raw_audio/transcripts.json")
OUT_ROOT = Path("data/raw_audio")
PIPER_VOICE_DIR = Path("data/tts_voices/piper")
PROVENANCE = Path("data/raw_audio/tts_provenance.json")

# ---------------------------------------------------------------- voice choice
# Kokoro ships 11 af_ (American female) and 9 am_ (American male) voices. We take
# two of each, spread across the roster rather than clustered, to get four
# distinct synthetic identities inside one generator. American English only, to
# match LibriSpeech's accent -- an accent mismatch between the real and fake
# sides would be its own shortcut. af_heart is retained because it is the voice
# Kokoro's own error message names as the example, i.e. its reference voice.
KOKORO_VOICES = ["af_heart", "af_sarah", "am_adam", "am_puck"]
KOKORO_LANG = "a"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SR = 24000  # confirmed in kokoro/__main__.py, pipeline.py, istftnet.py

# Piper: four en_US "medium"-tier voices trained on four DIFFERENT corpora, so
# the four identities are not four checkpoints of one dataset.
#
# en_US-libritts_r-medium is DELIBERATELY EXCLUDED. LibriTTS-R is derived from
# LibriSpeech, which is the corpus our bonafide clips come from. Training a
# spoof voice on the same source material as the real side puts the same
# speakers on both sides of the real/fake boundary, and any cross-synthesizer
# number measured against it would be contaminated at the identity level.
PIPER_VOICES = [
    "en_US-lessac-medium",   # Lessac Blizzard 2013 (CSTR)
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_US-joe-medium",
]

# ------------------------------------------------------------- sample rate
# Native rates are Kokoro 24000, Piper 22050, LibriSpeech 16000. Writing each at
# its native rate would make audio BANDWIDTH a perfect separator -- a 16 kHz
# file holds nothing above 8 kHz while a 24 kHz file holds content to 12 kHz --
# and the detector would score ~100% off the Nyquist ceiling, then collapse on
# the first 24 kHz authentic recording. Both classes must carry the identical
# channel distribution, so everything is resampled to 16 kHz.
TARGET_SR = 16000

TEST_SENTENCE = "The quick brown fox jumps over the lazy dog."


def normalise_text(raw: str) -> str:
    """LibriSpeech transcripts are ALL CAPS with no terminal punctuation.

    Fed verbatim, some front-ends read capitals as spelled-out letters and all
    of them lose sentence-final prosody with no full stop. Applied IDENTICALLY
    to every voice and both engines, so it cannot become a shortcut.
    """
    t = raw.strip()
    if not t:
        raise ValueError("empty transcript")
    t = t[0].upper() + t[1:].lower()
    if t[-1] not in ".?!":
        t += "."
    return t


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Polyphase resample to TARGET_SR at an exact integer ratio."""
    if sr == TARGET_SR:
        return x
    g = math.gcd(int(sr), TARGET_SR)
    return resample_poly(x, TARGET_SR // g, int(sr) // g)


def write_wav(path: Path, x: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite samples for {path}")
    if float(np.max(np.abs(x))) < 1e-4:
        raise ValueError(f"near-silent synthesis for {path}")
    sf.write(path, x, sr, subtype="PCM_16")
    return len(x) / sr


# ------------------------------------------------------------------- engines
def kokoro_synth(pipe, voice: str, text: str) -> tuple[np.ndarray, int]:
    """Kokoro yields Result objects carrying a 1-D float32 torch.Tensor.

    Long inputs can yield several segments, so concatenate rather than taking
    the first and silently truncating the sentence.
    """
    parts = []
    for res in pipe(text, voice=voice):
        a = res.audio
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        parts.append(np.asarray(a, dtype=np.float32).reshape(-1))
    if not parts:
        raise RuntimeError(f"kokoro produced no audio for voice {voice}")
    return np.concatenate(parts), KOKORO_SR


def piper_synth(voice_obj, text: str) -> tuple[np.ndarray, int]:
    """Piper yields AudioChunk objects with .audio_float_array and .sample_rate."""
    parts, rates = [], set()
    for chunk in voice_obj.synthesize(text):
        parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
        rates.add(int(chunk.sample_rate))
    if not parts:
        raise RuntimeError("piper produced no audio")
    if len(rates) != 1:
        raise RuntimeError(f"piper returned mixed sample rates: {rates}")
    return np.concatenate(parts), rates.pop()


def load_piper_voices() -> dict:
    from piper import PiperVoice
    from piper.download_voices import download_voice

    PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in PIPER_VOICES:
        onnx = PIPER_VOICE_DIR / f"{name}.onnx"
        if not onnx.exists():
            print(f"  downloading piper voice {name} ...")
            download_voice(name, PIPER_VOICE_DIR)
        v = PiperVoice.load(onnx)
        cfg = json.loads(Path(f"{onnx}.json").read_text(encoding="utf-8"))
        out[name] = dict(voice=v, onnx=onnx,
                         dataset=cfg.get("dataset"),
                         sr=cfg.get("audio", {}).get("sample_rate"),
                         num_speakers=cfg.get("num_speakers"))
    return out


def probe(pipe, piper_map) -> None:
    """Print what each engine ACTUALLY returns before any save logic runs."""
    print("=" * 72)
    print("API PROBE -- one test sentence through each engine")
    print("=" * 72)
    res = next(iter(pipe(TEST_SENTENCE, voice=KOKORO_VOICES[0])))
    a = res.audio
    print(f"kokoro: yields {type(res).__name__}; .audio is {type(a).__name__} "
          f"shape={tuple(a.shape)} dtype={a.dtype} -> in-memory tensor, sr={KOKORO_SR}")
    first = piper_map[PIPER_VOICES[0]]["voice"]
    chunks = list(first.synthesize(TEST_SENTENCE))
    c = chunks[0]
    print(f"piper : yields {len(chunks)} {type(c).__name__}; .audio_float_array is "
          f"{type(c.audio_float_array).__name__} shape={c.audio_float_array.shape} "
          f"dtype={c.audio_float_array.dtype} -> in-memory, sr={c.sample_rate}")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap transcripts (debug)")
    args = ap.parse_args()

    if not TRANSCRIPTS.exists():
        sys.exit(f"{TRANSCRIPTS} missing; run scripts/fetch_transcripts.py first")
    transcripts: dict[str, str] = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    items = sorted(transcripts.items())
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} transcripts loaded from {TRANSCRIPTS}\n")

    print("loading engines ...")
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=KOKORO_LANG, repo_id=KOKORO_REPO)
    piper_map = load_piper_voices()
    print()
    probe(pipe, piper_map)

    print("piper voice metadata (read from each .onnx.json, not assumed):")
    for n, m in piper_map.items():
        print(f"  {n:26s} dataset={str(m['dataset']):12s} sr={m['sr']} "
              f"num_speakers={m['num_speakers']}")
    print()

    # source_id is the ENGINE'S VOICE, not the LibriSpeech speaker whose sentence
    # we borrowed. group_split's leakage guard exists to stop the same underlying
    # VOICE landing on both sides of a split; for synthetic audio the voice is the
    # TTS preset. Inheriting the LibriSpeech speaker_id would assert an identity
    # these clips do not have.
    plan = []  # (generator, source_id, voice_name, utterance_id, text)
    for i, (uid, raw) in enumerate(items):
        kv = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        pv = PIPER_VOICES[i % len(PIPER_VOICES)]
        plan.append(("kokoro", f"kokoro_{kv}", kv, uid, raw))
        plan.append(("piper", f"piper_{pv}", pv, uid, raw))

    print("round-robin assignment (each transcript goes to ONE voice per engine):")
    for eng, voices in (("kokoro", KOKORO_VOICES), ("piper", PIPER_VOICES)):
        counts = {v: sum(1 for p in plan if p[0] == eng and p[2] == v) for v in voices}
        print(f"  {eng:7s} " + "  ".join(f"{v}={n}" for v, n in counts.items()))
    print()

    prov: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    t_start = time.time()

    for gen_name in ("kokoro", "piper"):
        rows = [p for p in plan if p[0] == gen_name]
        wrote = skipped = 0
        dur_total = 0.0
        t0 = time.time()
        for _, sid, vname, uid, raw in rows:
            dest = OUT_ROOT / gen_name / sid / f"{uid}.wav"
            if dest.exists():
                skipped += 1
                dur = sf.info(str(dest)).duration
            else:
                text = normalise_text(raw)
                if gen_name == "kokoro":
                    audio, native_sr = kokoro_synth(pipe, vname, text)
                else:
                    audio, native_sr = piper_synth(piper_map[vname]["voice"], text)
                dur = write_wav(dest, to_16k(audio, native_sr), TARGET_SR)
                wrote += 1
                if wrote % 10 == 0:
                    print(f"  {gen_name}: {wrote} written ...")
            dur_total += dur
            native_sr = KOKORO_SR if gen_name == "kokoro" else piper_map[vname]["sr"]
            # Rich provenance: the operation and its settings, not just a class.
            prov[f"{gen_name}/{sid}/{uid}"] = dict(
                engine=gen_name, voice=vname, source_id=sid, utterance_id=uid,
                native_sr=native_sr, output_sr=TARGET_SR,
                resampled=native_sr != TARGET_SR,
                text_normalisation="caps->sentence-case + terminal period",
                text=normalise_text(raw), duration_s=round(dur, 3),
                dataset=(piper_map[vname]["dataset"] if gen_name == "piper" else "kokoro-82M"),
            )
        stats[gen_name] = dict(wrote=wrote, skipped=skipped, clips=wrote + skipped,
                               dur=dur_total, secs=time.time() - t0,
                               voices=KOKORO_VOICES if gen_name == "kokoro" else PIPER_VOICES)
        print(f"  {gen_name}: {wrote} written, {skipped} already present "
              f"({time.time() - t0:.1f}s)")

    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nprovenance -> {PROVENANCE} ({len(prov)} entries, rebuilt from plan)")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, s in stats.items():
        print(f"{name:8s} clips={s['clips']:3d}  total={s['dur'] / 60:5.2f} min  "
              f"({s['secs']:.1f}s)  voices={len(s['voices'])}")
        for v in s["voices"]:
            sid = f"{name}_{v}"
            n = len(list((OUT_ROOT / name / sid).glob("*.wav")))
            d = sum(sf.info(str(p)).duration for p in (OUT_ROOT / name / sid).glob("*.wav"))
            print(f"           {sid:34s} {n:3d} clips  {d / 60:5.2f} min")
    tot = sum(s["clips"] for s in stats.values())
    print(f"{'TOTAL':8s} clips={tot:3d}  total="
          f"{sum(s['dur'] for s in stats.values()) / 60:5.2f} min  "
          f"({time.time() - t_start:.1f}s wall)")
    print(f"\nsource_id cardinality is now {len(KOKORO_VOICES)} per generator, so each")
    print("generator can be distributed across splits instead of moving as one block.")


if __name__ == "__main__":
    main()
