#!/usr/bin/env bash
# Orchestrates the corrected LoRA experiment on the GPU box, end to end.
# Every step gates on the previous one; the log tells the whole story.
set -euo pipefail
trap 'echo "EXPERIMENT_FAILED at line $LINENO"' ERR

cd "$(dirname "$0")"
T0=$SECONDS

echo "=== [1/8] venv + deps ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip wheel
# torch first, explicitly, so the CUDA build is a deliberate step not a
# transitive accident (Linux PyPI wheels bundle CUDA 12.x runtime)
pip install -q torch torchvision
python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("cuda available:", ok, "|", torch.cuda.get_device_name(0) if ok else "-")
raise SystemExit(0 if ok else 1)
EOF
pip install -q -r requirements.txt
echo "deps done at ${SECONDS}s"

echo "=== [2/8] test suite (pre) ==="
python -m tests.test_all > /dev/null && echo "tests exit 0"

echo "=== [3/8] sanity gate: head-only control at TrainConfig.head_lr ==="
python scripts/control_head_only.py

echo "=== [4/8] corrected LoRA end-to-end (train_lora_probe_v2) ==="
python scripts/train_lora_probe_v2.py

echo "=== [5/8] sklearn on LoRA-adapted features + FINAL TABLE ==="
python scripts/probe_adapted_features.py

echo "=== [6/8] test suite (post) ==="
python -m tests.test_all > /dev/null && echo "tests exit 0"

echo "=== [7/8] metrics json ==="
cat data/gpu_run_metrics.json

echo "=== [8/8] done ==="
echo "TOTAL_WALL_SECONDS=$((SECONDS - T0))"
echo "EXPERIMENT_DONE"
