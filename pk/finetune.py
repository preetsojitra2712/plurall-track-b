"""
Post-training / adaptation for a detection backbone.

The published recipe this follows (WACV 2026, "Unlocking the Hidden Potential of
CLIP in Generalizable Deepfake Detection" and the GenD / LNCLIP-DF line of work):

  backbone   CLIP ViT-L/14 visual encoder, frozen
  PEFT       LN-tuning -- train ONLY LayerNorm affine params, ~0.03% of weights
             (LoRA r=16 is the common alternative; NTIRE 2026 top entries used it)
  regularise L2-normalise features onto a hypersphere + metric learning
  preprocess face detect -> align -> crop 256x256 -> CLIP transform to 224x224
             (MTCNN with ~1.3x box expansion is the usual detector)
  reported   96.62 AUROC on Celeb-DF-v2, 98.0 on DFD, cross-dataset

Why PEFT rather than full fine-tuning: full FT on a narrow deepfake set
overwrites the general visual prior with generator-specific shortcuts. PEFT in a
low-data regime IS regularisation.

Adaptation to a NEW generator: roughly 700 samples gets you >98% in-domain with
almost any PEFT method. The hard part is not forgetting the old ones -- LoRA
adapters + REHEARSAL (replay a buffer of prior generators) is the reported best
combination for cross-domain stability. RehearsalBuffer below implements it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except Exception:                                     # keep the module importable
    TORCH = False
    torch = nn = F = None

__all__ = ["freeze_all_but_layernorm", "trainable_report", "LoRALinear",
           "inject_lora", "RehearsalBuffer", "HypersphereHead", "TrainConfig",
           "count_params"]


@dataclass
class TrainConfig:
    """Defaults that are sane for PEFT on a frozen vision backbone."""
    lr: float = 1e-4              # LN-tuning tolerates a higher LR than full FT
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 64
    warmup_frac: float = 0.1
    label_smoothing: float = 0.0  # OFF by default: it helps generalisation but
                                  # DEGRADES calibration, and calibration is the product
    rehearsal_ratio: float = 0.5  # half of each batch replayed from prior generators
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 0


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def freeze_all_but_layernorm(model, also_train: tuple = ("head", "classifier", "fc")):
    """LN-tuning. Freeze everything, then unfreeze LayerNorm affine params + the head.

    Returns (trainable, total, ratio). Expect ratio ~1e-4 for a ViT-L -- print it
    and say it out loud; it is the single most surprising and persuasive number
    in this recipe.
    """
    if not TORCH:
        raise RuntimeError("torch not available")
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad = True
    for name, p in model.named_parameters():
        if any(k in name for k in also_train):
            p.requires_grad = True
    tr, tot = count_params(model)
    return tr, tot, tr / max(tot, 1)


def trainable_report(model, top: int = 12) -> str:
    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    rows.sort(key=lambda x: -x[1])
    tr, tot = count_params(model)
    head = "\n".join(f"  {n:60s} {n_:>10,}" for n, n_ in rows[:top])
    return (f"trainable {tr:,} / {tot:,}  ({100 * tr / max(tot,1):.4f}%)\n"
            f"{head}\n  ... {max(0, len(rows)-top)} more tensors")


class LoRALinear(nn.Module if TORCH else object):
    """Low-rank adapter around a frozen nn.Linear: W x + (alpha/r) * B A x.

    A is init'd normal, B is init'd ZERO, so the adapter is an exact no-op at
    step 0 -- the model starts identical to the frozen base. That property is why
    LoRA is safe to bolt onto a production checkpoint.
    """

    def __init__(self, base, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        if not TORCH:
            raise RuntimeError("torch not available")
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scaling

    @torch.no_grad() if TORCH else (lambda f: f)
    def merge(self):
        """Fold the adapter into the base weight for inference -- zero latency cost."""
        self.base.weight += (self.B @ self.A) * self.scaling
        self.A.zero_(); self.B.zero_()
        return self.base


def inject_lora(model, target_substrings=("q_proj", "v_proj", "qkv"),
                r: int = 16, alpha: int = 32) -> int:
    """Replace matching nn.Linear modules with LoRALinear. Returns the count.

    Targeting query and value projections (not key, not MLP) is the standard
    cost/benefit point and comes straight from the LoRA paper.
    """
    if not TORCH:
        raise RuntimeError("torch not available")
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_substrings):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    for n, p in model.named_parameters():
        p.requires_grad = ("lora" in n.lower()) or n.endswith(".A") or n.endswith(".B")
    return replaced


class HypersphereHead(nn.Module if TORCH else object):
    """L2-normalise features onto a hypersphere, then a cosine classifier with a
    learnable scale. This is the regularisation the WACV 2026 work credits for
    cross-dataset generalisation, alongside the metric-learning term."""

    def __init__(self, in_dim: int, n_classes: int = 2, scale: float = 16.0):
        if not TORCH:
            raise RuntimeError("torch not available")
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, feats):
        f = F.normalize(feats, dim=-1)
        w = F.normalize(self.W, dim=-1)
        return self.scale * (f @ w.T)


class RehearsalBuffer:
    """Replay buffer for adapting to a new generator without forgetting old ones.

    Naive fine-tuning on a new generator degrades every previous one. Mixing a
    fixed fraction of prior-generator samples into every batch is the cheap,
    reliable fix, and it is what the cross-domain literature reports as the best
    pairing with LoRA adapters.

    Class-balanced reservoir sampling per generator so no single large family
    swamps the buffer.
    """

    def __init__(self, capacity_per_generator: int = 500, seed: int = 0):
        self.cap = capacity_per_generator
        self.rng = np.random.default_rng(seed)
        self.store: dict[str, list] = {}
        self.seen: dict[str, int] = {}

    def add(self, generator: str, item):
        buf = self.store.setdefault(generator, [])
        self.seen[generator] = self.seen.get(generator, 0) + 1
        if len(buf) < self.cap:
            buf.append(item)
        else:                                   # reservoir: uniform over the stream
            j = int(self.rng.integers(0, self.seen[generator]))
            if j < self.cap:
                buf[j] = item

    def extend(self, generator: str, items):
        for it in items:
            self.add(generator, it)

    def sample(self, n: int, exclude: str | None = None) -> list:
        gens = [g for g in self.store if g != exclude and self.store[g]]
        if not gens or n <= 0:
            return []
        per = max(1, n // len(gens))            # equal share per generator
        out = []
        for g in gens:
            buf = self.store[g]
            idx = self.rng.choice(len(buf), size=min(per, len(buf)), replace=False)
            out.extend(buf[i] for i in idx)
        self.rng.shuffle(out)
        return out[:n]

    def mixed_batch(self, new_items: list, ratio: float = 0.5) -> list:
        """Build a batch that is `ratio` replay, `1-ratio` new-generator data."""
        n_replay = int(len(new_items) * ratio / max(1e-9, 1 - ratio))
        batch = list(new_items) + self.sample(n_replay)
        self.rng.shuffle(batch)
        return batch

    def stats(self) -> dict:
        return {g: len(v) for g, v in sorted(self.store.items())}


def cosine_schedule(step: int, total: int, warmup_frac: float = 0.1) -> float:
    """LR multiplier. Warmup matters for PEFT because the adapters start at zero."""
    w = max(1, int(total * warmup_frac))
    if step < w:
        return step / w
    prog = (step - w) / max(1, total - w)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
