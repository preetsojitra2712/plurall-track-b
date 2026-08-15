# Data sources

Every source here needs a license that can be stated in one line, and every
assumption made about a dataset's composition is written down rather than
carried in someone's head.

---

## LibriSpeech ASR (dev-clean) — authentic speech

| field | value |
|---|---|
| Dataset name | LibriSpeech ASR corpus |
| HF repo id | `openslr/librispeech_asr` |
| Config / split used | `clean` / `validation` (i.e. dev-clean) |
| License (metadata tag, verbatim) | `cc-by-4.0` |
| License (Licensing Information section, verbatim) | `CC BY 4.0` — https://creativecommons.org/licenses/by/4.0/ |
| One-line statement | Creative Commons Attribution 4.0 International; redistribution and derivatives permitted with attribution. |
| Retrieval method | Streamed via the HuggingFace `datasets` library (`streaming=True`); no full-archive download. |
| Retrieved on | 2026-08-14 |
| Script | `scripts/fetch_librispeech_sample.py` |
| Landed at | `data/raw_audio/authentic/<speaker_id>/<utterance_id>.wav` |

**Citation**

```
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```

### What was pulled

10 speakers × 5 utterances = 50 clips, 16 kHz mono, 5.35 min total, 10.27 MB.
Clips were bounded to 2.0–12.0 s; anything outside that window was passed over.
The stream was scanned in order and the first 10 distinct `speaker_id` values
encountered were taken.

### Assumptions about composition — stated, not verified

These are assumptions this sample *makes*, and each one is a place the eval
could mislead if it turns out to be wrong:

1. **`speaker_id` is a genuine speaker-level identifier**, so grouping by it is
   sufficient to prevent the same voice landing on both sides of a split. Not
   independently verified against LibriSpeech's speaker metadata.
2. **Speakers are disjoint from the training material** any downstream
   synthesizer was built on. LibriSpeech is widely used as TTS training data —
   several public models are trained on `train-clean-360` — so dev-clean
   speakers are *probably* unseen by those, but this is an assumption, not a
   guarantee. If a cloned-voice positive is later generated from a model trained
   on LibriSpeech, the bonafide/spoof boundary is contaminated at the identity level.
3. **Read audiobook speech is not representative of the deployment channel.**
   It is clean, studio-adjacent, native-accented, fluent, and prompted. It
   under-represents the "unusual characteristics" band (heavy accents, stutters,
   atypical prosody) that false positives harm most, and it contains no
   telephone-channel material at all. Channel degradation must be *added*
   (`pk/degrade.py`); it is not present in the source.
4. **Taking the first 10 speakers in stream order is not a random sample** of
   dev-clean speakers. Whatever ordering the parquet shards impose is inherited.
   Fine for a smoke-scale bonafide set; not fine for any claim about
   speaker-demographic coverage.
5. **Utterance duration was filtered to 2–12 s**, which biases away from both
   very short utterances and long reads. Short-utterance behaviour is
   deliberately untested by this sample.

### Provenance labelling

These clips are **fully authentic / straight capture** in the Track B scenario
space: no benign processing, no enhancement, no codec transcoding beyond the
original corpus encoding. That matters because a denoised or enhanced authentic
recording is *not* a clone, and the label has to record the operation, not just
the binary class. Any processing applied downstream must be recorded as a
separate operation against these files rather than overwriting them in place.

**Original corpus encoding note:** LibriSpeech is distributed as FLAC (lossless)
sourced from MP3-encoded LibriVox audiobooks, so a lossy generation already sits
upstream of "native". These files were decoded from FLAC and written to
16-bit PCM WAV at the native 16 kHz — no resampling.

---

## Kokoro — generated positives (TTS)

| field | value |
|---|---|
| Package | `kokoro` |
| Version installed | 0.9.4 |
| Package license (classifier, verbatim) | `License :: OSI Approved :: Apache Software License` |
| Package license (metadata field) | full Apache License 2.0 text |
| One-line statement | Apache License 2.0; permissive, commercial use permitted with attribution and NOTICE retention. |
| Source repo | https://github.com/hexgrad/kokoro |
| Model weights | `hexgrad/Kokoro-82M` on HuggingFace |
| **Weights license (HF tag, verbatim)** | `apache-2.0` |
| Architecture | StyleTTS 2 + ISTFTNet decoder, 82M params |
| Voices used | `af_heart`, `af_sarah`, `am_adam`, `am_puck` (4 of 54) |
| Clips per voice | 13 / 13 / 12 / 12 (round-robin over the 50 transcripts) |
| Native sample rate | 24000 Hz |
| Landed at | `data/raw_audio/kokoro/kokoro_<voice>/<utterance_id>.wav` |
| Retrieved / generated on | 2026-08-14 |

License verified against the installed 0.9.4 distribution metadata and the model
repo tag separately — the package and the weights are distinct artifacts and are
recorded here as such.

**Voice choice rationale.** Kokoro has no default voice: `KPipeline.__call__`
raises `ValueError('Specify a voice: ...voice="af_heart"')` when `voice is None`,
so `af_heart` — the voice its own error message names — is retained as the
reference. The other three are chosen as 2 female + 2 male American voices
spread across the roster (11 `af_`, 9 `am_` available) rather than clustered.
American English only: an accent mismatch between the real and fake sides would
itself be a shortcut.

---

## Piper — generated positives (TTS)

| field | value |
|---|---|
| Package | `piper-tts` |
| Version installed | 1.6.1 |
| **Package license (metadata field, verbatim)** | `GPL-3.0-or-later` |
| One-line statement | GNU GPL v3.0 or later; copyleft — distributing a derived work obliges you to release source under GPL. |
| Source repo | http://github.com/OHF-voice/piper1-gpl |
| Voice models used | `en_US-lessac-medium`, `en_US-amy-medium`, `en_US-ryan-medium`, `en_US-joe-medium` |
| `dataset` field of each (read from `.onnx.json`) | `lessac`, `amy`, `ryan`, `joe` — four distinct corpora |
| Clips per voice | 13 / 13 / 12 / 12 (round-robin over the 50 transcripts) |
| Voice models stored at | `data/tts_voices/piper/` |
| Voice repo | `rhasspy/piper-voices` on HuggingFace |
| Voice repo license | MIT (repository-level) |
| **`lessac` training corpus** | Lessac Blizzard 2013 (CSTR, University of Edinburgh) |
| **`lessac` corpus license** | governed separately — https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html |
| Native sample rate | 22050 Hz (all four) |
| `num_speakers` | 1 (all four) |
| Landed at | `data/raw_audio/piper/piper_<voice>/<utterance_id>.wav` |
| Retrieved / generated on | 2026-08-14 |

**`en_US-libritts_r-medium` was deliberately EXCLUDED.** LibriTTS-R is derived
from LibriSpeech, which is the corpus the bonafide side comes from. A spoof
voice trained on the same source material puts the same speakers on both sides
of the real/fake boundary, and any cross-synthesizer number measured against it
would be contaminated at the identity level. This is the single most important
selection decision in the Piper set.

**Per-corpus licensing is NOT uniform across the four voices.** Only `lessac`
has been traced to its corpus licence here. `amy`, `ryan`, and `joe` carry the
repo-level MIT tag but their underlying corpora have not been individually
verified — do that before any redistribution.

**The licensing here is layered and the layers disagree — do not collapse them
to one line.** The *package* is GPL-3.0-or-later (this is the current
`OHF-voice/piper1-gpl` fork, not the older MIT-era Piper). The *voice repo* is
MIT at the repository level. The *training corpus* behind this specific voice,
Lessac Blizzard 2013, carries its own CSTR licence which must be read directly
before any redistribution or commercial use — it is not an OSI licence and is
not equivalent to the repo's MIT tag. For local research use this is fine; for
anything shipped, the corpus licence is the binding constraint.

**Voice choice rationale.** `en_US` matches the corpus accent; all four are the
`medium` quality tier so tier is held constant across identities; all four train
on different corpora so the four identities are not four checkpoints of one
dataset. ~160 voice models are available.

---

## Derived dataset properties — read before training on this

**Sample rate was forced to 16 kHz for every clip, including the synthetic ones.**
Native rates differ (LibriSpeech 16000, Piper 22050, Kokoro 24000) and audio
bandwidth follows the rate: a 16 kHz file holds nothing above 8 kHz while a
24 kHz file holds content to 12 kHz. Left native, the Nyquist ceiling alone
would separate bonafide from spoof perfectly, and the resulting detector would
collapse the first time it met a 24 kHz authentic recording. Resampling is
polyphase (`scipy.signal.resample_poly`) at exact integer ratios.

**Residual concern:** the synthetic clips carry *our* resampler's anti-alias
rolloff; the authentic clips carry whatever LibriSpeech's corpus builders used.
That is a second-order channel difference between the classes. Passing the
authentic side through an identical 16k→24k→16k round trip would equalise it,
and is worth doing before any result is believed.

**Text normalisation** applied identically to both engines: LibriSpeech
transcripts are ALL CAPS with no terminal punctuation, so each was lowered to
sentence case and given a full stop. Identical across engines, so it cannot act
as a cross-engine shortcut. Recorded per clip in `data/raw_audio/tts_provenance.json`.

**Level was equalised across the WHOLE tree — `scripts/normalize_loudness.py`.**
Piper peak-normalises its output to digital full scale while LibriSpeech sits
near half scale, and before this step peak amplitude alone separated authentic
from Piper at **1.000** balanced accuracy (RMS at 0.910). Every clip in every
generator, authentic included, is now normalised to **RMS 0.05**.

*Diagnosis first:* the full-scale peaks were checked for real clipping by
reading int16 and measuring runs at the rail. 25 of 28 affected clips had a
single-sample peak, 3 had none at exactly the rail, longest plateau was 1 sample.
**No flat-topped clipping** — Piper applies peak-normalisation gain, not
limiting, so nothing was distorted and the fix is pure gain.

*Why RMS, not LUFS:* RMS is exactly idempotent (re-running is a no-op), needs no
extra dependency, and directly zeroes the leaking axis. LUFS (BS.1770
K-weighting) is the better perceptual measure and would be right if the goal
were matching listener loudness. Target 0.05 was verified against the actual
tree before writing: max crest factor 14.45 puts the loudest resulting peak at
0.723, so nothing clips. Caveat: whole-clip RMS includes silence, so clips with
more leading/trailing silence end up with slightly hotter speech.

**Separability after the fix** (best single-threshold balanced accuracy vs
authentic; 0.5 = no information). Run `scripts/check_shortcuts.py` to reproduce:

| generator | peak | rms | duration | silence |
|---|---|---|---|---|
| kokoro (before) | 0.640 | 0.620 | 0.550 | 0.650 |
| kokoro (after) | 0.650 | 0.560 | 0.550 | 0.650 |
| piper (before) | **1.000** | **0.910** | 0.600 | 0.870 |
| piper (after) | 0.600 | 0.570 | 0.600 | **0.870** |

**Channel equalisation — `scripts/equalize_channel.py` (applied 2026-08-14).**
The 7–8 kHz band-energy confound (authentic 1.93e-04 vs kokoro 3.09e-03 /
piper 8.47e-04; balanced acc 0.910/0.710) was addressed at the source. A
16k→24k→16k resample round-trip on authentic was **measured first and rejected**:
it shifted the gap by −7.8% (widened it) — resampling cannot create energy the
source never had. Instead an RBJ high-shelf **cut** (f0 6.5 kHz, −12 dB, S=1,
single pass) was applied to the synthetic side only, logged per clip in
`channel_eq_log.json`, idempotent. −12 dB was chosen once from the coarse
kokoro/authentic power ratio (~16x), not iterated against the outcome; piper
consequently overshoots below authentic (6.68e-05) and this is reported, not
tuned away. This is channel *matching*, not asymmetric augmentation: the
authentic corpus already carries an upstream MP3-lineage rolloff, so the
synthetic side is passed through an approximation of the same channel. After:
kokoro 2.26e-04, piper 6.68e-05; separability on this axis 0.660/0.680.
Loudness was re-normalised afterwards (gains 1.00–1.01).

**The layer-probe curve was nearly unchanged by this fix** (layer 0 still 1.000
on both val and test) — so the low-layer separability was NOT primarily riding
the 7–8 kHz band, and either finer spectral-shape channel residue or genuine
shared vocoder texture remains. The band axis is now a permanent fifth check in
`check_shortcuts.py` regardless.

**Silence ratio is still leaky at 0.870 for Piper and is NOT fixed.** Piper's
median silent-frame fraction is 0.143 against authentic's 0.274: TTS does not
produce the leading and trailing room tone that a recorded utterance carries.
That is a *packaging* artifact, not a synthesis artifact — a detector leaning on
it is reading "was this trimmed", not "was this generated". The fix is to pad or
trim both classes to a common silence distribution, applied identically. Until
that is done, treat any headline number on this set as optimistic.

**Telephone-channel degraded eval set — `data/raw_audio_degraded/` (2026-08-14).**
A degraded COPY of the test split (clean files untouched), built by
`scripts/degrade_eval_set.py` with `pk/audio_degrade.py`, per-clip recipes in
`degrade_log.json`. Honesty disclosures, mirrored from the module docstring:

- **G.711 μ-law is implemented for real in numpy** (8 kHz resample + μ-law
  companding + 8-bit quantisation — that IS the codec, not an approximation).
- **AMR-NB and Opus are NOT included**: they need ffmpeg, which is not on this
  machine. Absent, not faked.
- **The RIR is SYNTHETIC** (exponentially-decaying filtered noise, rt60
  0.15–0.6 s): a standard cheap approximation that captures reverberant
  smearing statistics but no real room's geometry or early reflections. A
  measured RIR set (e.g. openSLR RIRS_NOISES) is the upgrade path.
- Packet-loss concealment is **silence** (hardest case); real endpoints
  interpolate.
- The identical parameter distribution was applied to both classes
  (10 authentic + 50 piper), per the shortcut rule.

**Identity diversity is now 4 voices per generator** (was 1). Kokoro:
`af_heart`, `af_sarah`, `am_adam`, `am_puck`. Piper: `lessac`, `amy`, `ryan`,
`joe`. This matters structurally: with one voice per generator, `source_id` had
cardinality 1 and `group_split` had to move each generator between splits as a
single indivisible block, which put all 100 positives in train and left test
with no positives at all.

**Split parameters are not the defaults, deliberately.** The manifest is built
with `--holdout piper --val-fraction 0.25`. Holding out an engine entirely is
what makes test a genuine *unseen-synthesizer* evaluation: train sees only
Kokoro (StyleTTS2), test sees only Piper (VITS). `val_fraction` 0.25 is the
smallest value that yields both classes in every split across all six seeds
tested — 0.15 and 0.20 are seed-dependent and produced a single-class val. The
underlying fragility is low `source_id` cardinality (10 speakers + 4 + 4 voices);
`group_split` carves val by source hash without stratifying on label, so with
few groups a single-class val is easy to hit. More voices per engine is the
durable fix.
