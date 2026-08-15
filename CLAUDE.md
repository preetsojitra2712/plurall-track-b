# CLAUDE.md — project context for Claude Code

## What this repo is
A working toolkit for deepfake-detection ML: evaluation harness, leakage-safe
dataset splitting, a compound degradation engine, calibration, six-dimension
late fusion, and PEFT adaptation helpers. Written for a trust-and-safety
detection platform where a verdict has to be explainable and defensible.

## Non-negotiable conventions — follow these without being asked

**Metrics.** Never report accuracy. The headline metric is `tpr_at_fpr` at a
stated FPR (default 1%), reported PER GENERATOR, never only as an aggregate.
Also report partial AUC in the low-FPR region, EER for audio, and ECE/Brier for
calibration. Every point estimate that will be compared against another gets a
bootstrap CI; for A/B use `paired_bootstrap_delta`, not two independent CIs.

**Splits.** Never `train_test_split` on rows. Split by `generator` AND by
`source_id` using `pk.data.group_split`, then call `pk.data.leakage_report(...,
strict=True)`. A random split measures memorisation. If you write a split, you
write the leakage assertion in the same commit.

**Calibration.** Raw model outputs are not probabilities. Calibrate before
fusing and before reporting anything as `confidence`. Fit per segment
(media type × quality bucket) via `SegmentedCalibrator`, with a global fallback.
Temperature scaling is the default because it is monotone and cannot overfit.

**Fusion.** Late fusion in logit space with an availability mask. Never zero-fill
a missing dimension — skip it. Keep the head linear unless there is evidence it
must be non-linear, because `w_d * logit(p_d)` is the number printed on the
evidence card. Train with modality dropout.

**Augmentation.** Model the distribution channel, not generic vision augs:
JPEG recompression, rescale, platform re-encode, screenshot simulation, codec
simulation for audio. Apply the IDENTICAL distribution to both classes or you
have built a shortcut. Log the recipe string for every degraded sample.

**Fine-tuning.** Prefer PEFT (LayerNorm-tuning ~0.03% of params, or LoRA r=16 on
q/v projections) over full fine-tuning. Full FT on a narrow deepfake set
overwrites the general visual prior with generator-specific shortcuts. When
adapting to a NEW generator, always mix in replay from `RehearsalBuffer` —
naive fine-tuning forgets prior generators.

## Style
- Python 3.10+, type hints on public functions, no framework beyond
  numpy/pandas/sklearn/torch.
- Small pure functions. No hidden global state.
- Every new metric or transform gets a test in `tests/test_all.py` with a real
  assertion (compare against sklearn where one exists).
- Docstrings say WHY, not what. Explain the failure mode the code prevents.
- Fail loudly on bad data. Never silently impute NaN scores.

## Layout
```
pk/metrics.py    TPR@FPR, pAUC, EER, ECE, Brier, bootstrap CIs
pk/calibrate.py  temperature / Platt / isotonic, segmented
pk/data.py       manifest schema, group_split, leakage_report, phash dedup
pk/degrade.py    compound degradation engine + JPEG sweep
pk/fusion.py     six-dimension late fusion, evidence cards, abstention
pk/harness.py    evaluate(), release_gate(), to_markdown()
pk/finetune.py   LN-tuning, LoRA injection, RehearsalBuffer
scripts/         smoke_test.py, run_eval.py, make_manifest.py
tests/test_all.py  76 assertions, run with `python -m tests.test_all`
```

## Commands
```
python -m tests.test_all          # full test suite, no pytest needed
python scripts/smoke_test.py      # 30s end-to-end demo on synthetic data
python scripts/run_eval.py --scores s.csv --baseline reports/prod.json --out reports/new
python scripts/make_manifest.py --root /data --holdout sdxl --out manifest.csv
```

## When I ask you to add something
1. Ask what the failure mode is before writing code, if it is not obvious.
2. Write the test first or alongside — an assertion, not a print.
3. Run `python -m tests.test_all` before telling me it works.
4. If a change would alter an existing metric's value, say so explicitly.
