# setup.ps1 — Windows PowerShell version of setup.sh
# Run this FROM INSIDE the plurall-kit folder, not from System32 or your home dir.
#
# Usage:
#   .\setup.ps1
# If PowerShell blocks it ("running scripts is disabled on this system"), run:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

python -m venv .venv
$py = ".\.venv\Scripts\python.exe"   # call the venv's python directly by path —
                                      # sidesteps the PowerShell execution-policy
                                      # error that Activate.ps1 often hits

& $py -m pip install -U pip wheel

Write-Host "--- core (fast) ---"
& $py -m pip install numpy scipy pandas scikit-learn pillow matplotlib tqdm pyyaml

Write-Host "--- torch (slow, ~2GB) ---"
& $py -m pip install torch torchvision

Write-Host "--- vision stack ---"
& $py -m pip install timm transformers peft open_clip_torch opencv-python-headless facenet-pytorch

Write-Host "--- pre-caching model weights so you are not downloading on their wifi ---"
$cache = @'
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
'@
$cache | & $py -

Write-Host "--- verifying ---"
& $py -m tests.test_all
& $py scripts\smoke_test.py

Write-Host ""
Write-Host "READY."
Write-Host "For interactive work later, activate the venv with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "If THAT errors with 'running scripts is disabled', run this once (current window only):"
Write-Host "    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
Write-Host "Weights are cached under $env:USERPROFILE\.cache\huggingface and $env:USERPROFILE\.cache\torch"
