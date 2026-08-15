# Cost log

Costs are tracked as wall-clock and instance uptime, not just compute — boot,
setup, and idle time all bill the same as training.

## GPU cost — Lambda `gpu_1x_h100_sxm5` (68.209.73.154), 2026-08-14

| field | value |
|---|---|
| Instance boot (`uptime -s`, UTC) | 2026-08-14 19:40:21 |
| Terminated (UTC, from Lambda billing) | 2026-08-14 21:00 |
| Billable uptime (Lambda billing) | **1.33 hr** (80 min) |
| Rate (actual) | **$4.29 / hr** |
| **Total spend (actual)** | **$5.69** |
| Region | Georgia, USA (us-southeast-1) |

### What the uptime bought (source of truth: `data/gpu_run_logs/`)

| run | wall | outcome |
|---|---|---|
| attempt 1 (`run_attempt1.log`) | ~2 min (mostly pip) | FAILED — Windows backslash paths in manifest vs Linux; fixed and reshipped |
| attempt 2 (`run_attempt2_gatefail.log`) | ~3 min | GATE FAIL at 8 epochs, by design — budget ended mid-descent (loss 0.649, cosine anneal) |
| gate standalone, 100 epochs | ~80 s | GATE PASS: val 1.000 / test 0.914 |
| full corrected run (`run.log`) | **103 s total; LoRA training 71.8 s** | EXPERIMENT_DONE — final table produced, tests exit 0 pre+post |

Productive GPU compute ≈ 4 min of the 80 billed minutes (~5%). The other ~95%
— boot, pip installs, transfers, debugging the path bug, and idle time between
attempts and before termination — cost ~$5.40 of the $5.69. The honest
conclusion for a run this small: instance management, not compute, is the cost
driver. The training that produced the headline −0.102 result cost about
**$0.09** of H100 time (71.8 s at $4.29/hr).

### Comparison: the same experiment family on CPU (local, no $ cost)

| run | wall |
|---|---|
| LoRA v1, 8 epochs, CPU (undertrained — superseded) | 727 s |
| feature extraction, 150 clips × 7 layers (run twice: pre/post channel-eq) | 173 s + 134 s |
| layer probes, sklearn (7 layers × fit+eval) | seconds |

H100 did 12.5× the epochs of the CPU v1 run in 1/10 the wall time (71.8 s vs
727 s), i.e. ~125× effective throughput for this workload.

### Artifacts retrieved before termination

- `data/lora_v2_state.pt` (602,005 B — 24 LoRA tensors, 147,456 params, verified finite via torch.load)
- `data/gpu_run_metrics.json` (413 B — all three result rows)
- `data/gpu_run_logs/run.log`, `run_attempt1.log`, `run_attempt2_gatefail.log`

## Generation cost (TTS positives) — local CPU, no $ cost

| item | wall |
|---|---|
| Kokoro, 50 clips ×2 (single-voice run, then 4-voice regeneration) | 288 s + 399 s |
| Piper, 50 clips ×2 | 33 s + 34 s |
| LibriSpeech streaming pulls (50 clips + transcripts) | minutes, network-bound |

Kokoro is ~12× slower than Piper per clip on CPU — at scale-up, Piper volume
is nearly free while Kokoro becomes the generation budget.

## Human/agent review time

Not separately metered; the significant non-compute sinks were diagnosing the
undertrained-head confound (one control experiment), the corpus-bandwidth
confound (round-trip measurement + shelf design), and the Windows→Linux path
bug (one failed GPU attempt).
