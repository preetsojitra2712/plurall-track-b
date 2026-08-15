# DEMO — live walkthrough cheat sheet

All commands verified and timed on this machine, 2026-08-14, in the venv.
Rule used: anything measured over ~15 s is shown from its saved artifact, not
re-run live.

## Step 0 — pre-verified state (run before the room, not in it)

`python -m tests.test_all` → **ALL 92 CHECKS PASSED, exit 0** (measured 11.4 s).

## Setup (once, at the start)

```powershell
Set-Location "C:\Users\preet\Downloads\plurall-kit\plurall-kit"
.\.venv\Scripts\Activate.ps1        # verified: 0.44 s
# if that errors with "running scripts is disabled":
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## The narrative, in order

### 1. Data + provenance — SHOW, don't run
Open: `data/SOURCES.md`
> Cue: "Every clip has a license, a retrieval method, and a stated assumption — including the ones that could bite us."

### 2. Leakage-safe split — LIVE (0.9 s)
```powershell
python scripts\make_manifest.py --root data/raw_audio --out data/manifest.csv --holdout piper --val-fraction 0.25
```
> Cue: "Split by voice and generator, never by row — Piper is held out entirely, and the class-coverage check exists because a leakage-clean split can still be single-class."

### 3. Shortcut audit — LIVE (2.1 s)
```powershell
python scripts\check_shortcuts.py
```
> Cue: "Before any model: can a scalar alone separate the classes? Peak amplitude used to do it at 1.000 — we fixed loudness and bandwidth; silence is the one disclosed leak left."

### 4. Layer-wise probing — LIVE (3.6 s), then SHOW the curve
```powershell
python scripts\probe_layers.py
```
Open: `data/layer_probe_curve.png`
> Cue: "Signal peaks at the CNN output and dies with depth — mid-layers are the honest tap; layer-0 perfection is a documented open question, not a brag."

### 5. The confound we found — SHOW
Open: `data/SOURCES.md` (channel-equalisation section) and `data/results_summary.md` (pre- vs post-eq tables)
> Cue: "The corpus itself leaked bandwidth — we measured the proposed fix, rejected it with physics, applied a disclosed shelf instead, and the curve barely moved, which is itself a finding."

### 6. PEFT + the optimizer bug — SHOW
Open: `data/gpu_run_logs/run_attempt2_gatefail.log` (the gate doing its job), then `data/gpu_run_metrics.json`
> Cue: "The first LoRA result was fiction — an undertrained head, caught by a 5-second control; the gate refused to let us interpret the run until the recipe could train a head at all."

### 7. Corrected GPU result — SHOW
Open: `data/results_summary.md` (GPU run table) — frozen 0.970 vs sklearn-on-adapted 0.868
> Cue: "With the confound removed, adaptation genuinely degraded cross-synthesizer transfer by ten points — the classic trap, now properly earned, and the case for rehearsal."

### 8. Telephone-channel degradation — SHOW
Open: `data/results_summary.md` (clean-vs-telephone table); recipes in `data/raw_audio_degraded/degrade_log.json`
> Cue: "Real G.711 in numpy, disclosed synthetic RIR — the frozen probe collapses to a coin flip under the channel while the adapted model keeps most of its skill, so the clean ranking reverses."

### 9. Gaussian confidence — LIVE (3.6 s), then SHOW the figure
```powershell
python scripts\gaussian_confidence_report.py
```
Open: `data/gaussian_confidence.png`
> Cue: "The posterior says 'spoof, 1.000' — but typicality says Piper is 8 sigma from the fitted spoof Gaussian, so the honest output is ABSTAIN: that's the open-set answer for an engine like ElevenLabs we've never seen."

### 10. Summary + cost — SHOW
Open: `data/results_summary.md`, then `data/cost_log.md`
> Cue: "Every number in one place, straight from the artifacts — and the H100 bill was four minutes of compute; instance management, not training, was the real cost."

---

## Do NOT run live (and why)

| item | reason | show instead |
|---|---|---|
| `train_lora_probe*.py` | needs the terminated GPU box, or 12+ min CPU | `data/gpu_run_metrics.json`, `data/gpu_run_logs/run.log` |
| `extract_layer_features.py` | 130–170 s, touches HF cache | `data/features/w2v2_layers.npz` exists; curve PNG |
| `eval_degraded.py` | 2 wav2vec2 passes, ~4 min CPU | `data/degraded_scores.csv`, summary table |

## Verified timings (Measure-Command, this machine)

| command | wall | live? |
|---|---|---|
| `.\.venv\Scripts\Activate.ps1` | 0.44 s | yes |
| `python -m tests.test_all` | 11.4 s | yes (borderline — step 0 covers it if time is tight) |
| `python scripts\check_shortcuts.py` | 2.1 s | yes |
| `python scripts\probe_layers.py` | 3.6 s | yes |
| `python scripts\gaussian_confidence_report.py` | 3.6 s | yes |
| `python scripts\make_manifest.py --root data/raw_audio --out data/manifest.csv --holdout piper --val-fraction 0.25` | 0.9 s | yes |

All show-artifacts verified present and opening cleanly (PNGs render, JSON
parses with 3 runs, CSV has 120 rows, state dict loads) on 2026-08-14.
