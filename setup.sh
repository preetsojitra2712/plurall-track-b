#!/usr/bin/env bash
# One command to a working environment. Run BEFORE you travel, not at the desk.
# macOS / Linux. On Windows, use setup.ps1 instead.
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

echo "--- core (fast) ---"
pip install numpy scipy pandas scikit-learn pillow matplotlib tqdm pyyaml

echo "--- torch (slow, ~2GB) ---"
pip install torch torchvision

echo "--- vision stack ---"
pip install timm transformers peft open_clip_torch opencv-python-headless facenet-pytorch

echo "--- pre-caching model weights so you are not downloading on their wifi ---"
python - <<'PY'
import timm, torch
for name in ["vit_base_patch16_clip_224.openai", "efficientnet_b0", "resnet50"]:
    try:
        timm.create_model(name, pretrained=True)
        print("cached", name)
    except Exception as e:
        print("SKIP", name, e)
try:
    from transformers import CLIPModel, CLIPProcessor
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("cached CLIP ViT-B/32")
except Exception as e:
    print("SKIP clip", e)
PY

echo "--- verifying ---"
python -m tests.test_all | tail -3
python scripts/smoke_test.py | tail -5
echo
echo "READY. Weights are cached under ~/.cache/huggingface and ~/.cache/torch"
