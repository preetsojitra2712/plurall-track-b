"""
Compound degradation engine.

Why this file exists: the NTIRE 2026 Robust Deepfake Detection Challenge (337
participants) was built entirely around the finding that detection performance is
"nearly worthless in the real world if it suffers under exposure to even slight
image degradation" -- both accidental (platform re-encoding) and malicious
(laundering deliberately aimed at a detector's weak band). Top solutions trained
against randomised multi-operation degradation pipelines.

And Deepfake-Eval-2024 quantified the gap: open-source SOTA detectors lose ~45%
AUC on images, ~50% on video, ~48% on audio when moved from academic benchmarks
to media actually circulating online, with many landing near 0.5 -- chance.

So: augment against the distribution channel, not against generic vision augs.
CRITICAL: apply the identical degradation distribution to BOTH classes, or you
have just built a shortcut.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

__all__ = ["DegradationEngine", "jpeg", "rescale", "blur", "noise",
           "screenshot_sim", "QUALITY_BUCKETS", "quality_bucket"]

QUALITY_BUCKETS = ("native", "q_high", "q_mid", "q_low")


def _pil(img):
    return img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))


def jpeg(img, quality: int):
    """Re-encode. The single most important augmentation in this domain --
    it attacks exactly the high-frequency band spectral detectors live in."""
    buf = io.BytesIO()
    _pil(img).convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rescale(img, factor: float, resample=Image.BILINEAR):
    im = _pil(img)
    w, h = im.size
    small = im.resize((max(8, int(w * factor)), max(8, int(h * factor))), resample)
    return small.resize((w, h), resample)


def blur(img, radius: float):
    return _pil(img).filter(ImageFilter.GaussianBlur(radius=float(radius)))


def noise(img, sigma: float, rng=None):
    rng = rng or np.random.default_rng()
    a = np.asarray(_pil(img).convert("RGB"), dtype=np.float32)
    a = a + rng.normal(0, sigma, a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def contrast(img, f: float):
    return ImageEnhance.Contrast(_pil(img).convert("RGB")).enhance(float(f))


def saturation(img, f: float):
    return ImageEnhance.Color(_pil(img).convert("RGB")).enhance(float(f))


def screenshot_sim(img, rng=None):
    """Screen-capture-of-a-screen: mild moire, slight rescale, re-encode.

    Matters because a huge share of real reports arrive as a photo or capture of
    a screen, which destroys metadata and most spectral signal at once.
    """
    rng = rng or np.random.default_rng()
    im = _pil(img).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape[:2]
    period = rng.uniform(2.5, 5.0)
    grid = (np.sin(np.arange(w) * 2 * np.pi / period)[None, :, None] * rng.uniform(2, 6))
    a = np.clip(a + grid, 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    im = rescale(im, rng.uniform(0.7, 0.95))
    return jpeg(im, int(rng.integers(55, 85)))


class DegradationEngine:
    """Randomised compound pipeline: sample k ops from the menu and apply in order.

    Usage
    -----
    eng = DegradationEngine(seed=0)
    out, recipe = eng(img)                     # random severity
    out, recipe = eng(img, severity="heavy")   # forced band
    `recipe` is logged so every degraded sample stays reproducible -- that is the
    difference between an experiment and an anecdote.
    """

    SEVERITY = {
        "light": dict(k=(1, 2), jpeg_q=(80, 95), scale=(0.85, 1.0),
                      blur_r=(0.0, 0.5), noise_s=(0.0, 2.0)),
        "medium": dict(k=(2, 3), jpeg_q=(55, 80), scale=(0.6, 0.9),
                       blur_r=(0.3, 1.2), noise_s=(1.0, 5.0)),
        "heavy": dict(k=(3, 5), jpeg_q=(25, 55), scale=(0.35, 0.7),
                      blur_r=(0.8, 2.5), noise_s=(3.0, 10.0)),
    }

    def __init__(self, seed: int = 0, p_screenshot: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.p_screenshot = p_screenshot

    def __call__(self, img, severity: str | None = None):
        sev = severity or str(self.rng.choice(["light", "medium", "heavy"], p=[0.4, 0.4, 0.2]))
        cfg = self.SEVERITY[sev]
        im = _pil(img).convert("RGB")
        recipe = [f"severity={sev}"]

        if self.rng.random() < self.p_screenshot:
            im = screenshot_sim(im, self.rng)
            recipe.append("screenshot_sim")

        menu = ["jpeg", "rescale", "blur", "noise", "contrast", "saturation", "jpeg2"]
        k = int(self.rng.integers(cfg["k"][0], cfg["k"][1] + 1))
        ops = list(self.rng.choice(menu, size=min(k, len(menu)), replace=False))
        # a second JPEG pass at the end is the common real-world case (cyclic re-encode)
        if "jpeg2" in ops:
            ops = [o for o in ops if o != "jpeg2"] + ["jpeg"]

        for op in ops:
            if op == "jpeg":
                q = int(self.rng.integers(*cfg["jpeg_q"]))
                im = jpeg(im, q); recipe.append(f"jpeg(q={q})")
            elif op == "rescale":
                f = float(self.rng.uniform(*cfg["scale"]))
                im = rescale(im, f); recipe.append(f"rescale({f:.2f})")
            elif op == "blur":
                r = float(self.rng.uniform(*cfg["blur_r"]))
                if r > 0.05:
                    im = blur(im, r); recipe.append(f"blur({r:.2f})")
            elif op == "noise":
                s = float(self.rng.uniform(*cfg["noise_s"]))
                if s > 0.1:
                    im = noise(im, s, self.rng); recipe.append(f"noise({s:.1f})")
            elif op == "contrast":
                f = float(self.rng.uniform(0.8, 1.25))
                im = contrast(im, f); recipe.append(f"contrast({f:.2f})")
            elif op == "saturation":
                f = float(self.rng.uniform(0.8, 1.25))
                im = saturation(im, f); recipe.append(f"sat({f:.2f})")

        return im, "|".join(recipe)

    def sweep(self, img, qualities=(95, 85, 75, 65, 55, 45, 35, 25)):
        """Single-axis JPEG sweep for a robustness CURVE.

        Report recall-vs-JPEG-quality as a curve, never as a single number. The
        curve is what tells you whether a model survives contact with a platform.
        """
        return {q: jpeg(img, q) for q in qualities}


def quality_bucket(jpeg_quality_estimate: float | None) -> str:
    """Map an estimated compression level to a calibration segment key."""
    if jpeg_quality_estimate is None:
        return "native"
    q = float(jpeg_quality_estimate)
    if q >= 90:
        return "q_high"
    if q >= 65:
        return "q_mid"
    return "q_low"
