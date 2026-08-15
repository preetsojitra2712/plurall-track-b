# plurall-kit

A working toolkit for deepfake-detection ML work: evaluation harness,
leakage-safe dataset splitting, compound degradation, calibration, six-dimension
late fusion, and PEFT adaptation.

Built as interview preparation for an AI/ML Engineer round focused on
**deepfake model post-training, data processing, fine-tuning, and harness
engineering**. Everything here runs and is covered by real assertions.

```bash
# macOS / Linux
./setup.sh

# Windows (PowerShell)
.\setup.ps1
# if blocked: powershell -ExecutionPolicy Bypass -File .\setup.ps1

python -m tests.test_all         # 76 assertions
python scripts/smoke_test.py     # 30-second end-to-end demo
```

## Why each piece exists

| Module | The failure it prevents |
|---|---|
| `pk/metrics.py` | Reporting accuracy on a rare-positive problem, where "always authentic" scores 99.5% and catches nothing. Headline metric here is TPR at a fixed low FPR. |
| `pk/data.py` | Random splits. The same actor, source video, and generator land on both sides, so you measure memorisation and ship a model that collapses on the first new generator. |
| `pk/degrade.py` | Training on clean data and deploying into a world of WhatsApp forwards and screenshots. Deepfake-Eval-2024 measured open-source SOTA losing ~45–50% AUC moving from academic sets to media actually circulating online. |
| `pk/calibrate.py` | Shipping a raw softmax output as "confidence". Neural nets are systematically overconfident, and calibration is a customer-facing field in this product. |
| `pk/fusion.py` | Early fusion, which destroys per-dimension explainability and breaks the moment a modality is missing. |
| `pk/harness.py` | Aggregate metrics that hide total blindness to one popular new generator. |
| `pk/finetune.py` | Full fine-tuning that overwrites the general visual prior, and naive adaptation that forgets every previous generator. |

## The one output that matters

`harness.to_markdown()` produces a per-generator table sorted worst-first, a
robustness curve by degradation level, calibration by quality bucket, and an
explicit coverage-gap section. `harness.release_gate()` turns "should we promote
this model?" into a function that returns a boolean and a list of reasons, with
two gates: no overall regression, and no single generator dropping more than a
configured amount.

## Design commitments

1. **Split by generator and source, never at random.** Leakage assertions run in
   `leakage_report(strict=True)` and raise.
2. **Calibrate before you fuse, and fuse in logit space.** Log-odds add;
   probabilities do not.
3. **Availability mask, never zero-fill.** A missing EXIF card is `STRIPPED`,
   not 0.5.
4. **Keep the fusion head linear** while you can — the weights are what you print
   on the evidence card.
5. **Augment against the distribution channel**, identically for both classes.
6. **PEFT over full fine-tuning**, plus rehearsal when adapting to a new generator.

## References the code follows

- *Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection* /
  GenD (WACV 2026) — LayerNorm-tuning at 0.03% of parameters, L2-normalised
  hyperspherical features, and paired real/fake from the same source video.
- NTIRE 2026 Robust Deepfake Detection Challenge — randomised compound
  degradation as the core training strategy; DINOv2 + CLIP fusion with LoRA in
  top entries.
- Deepfake-Eval-2024 — the in-the-wild AUC collapse that motivates `degrade.py`.
- DeepfakeBench / DF40 — the standard preprocessing and evaluation protocol.
- ASVspoof 5 — EER and min-DCF as the audio metric convention.

## Platform notes

`setup.sh` is bash (macOS/Linux/WSL/Git Bash). `setup.ps1` is native Windows
PowerShell and avoids `Activate.ps1`'s execution-policy issues by calling the
venv's `python.exe` directly by path during setup; you only need to activate
the venv afterward for interactive work.
