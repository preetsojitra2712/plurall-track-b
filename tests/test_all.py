"""Run: python -m tests.test_all  (from the repo root). No pytest required."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image

from pk import metrics as M
from pk import calibrate as C
from pk import data as D
from pk import degrade as G
from pk import fusion as FU
from pk import harness as H

rng = np.random.default_rng(0)
PASS = []


def check(name, cond, extra=""):
    assert cond, f"FAILED: {name} {extra}"
    PASS.append(name)


# ------------------------------------------------------------------ metrics
from sklearn.metrics import roc_auc_score
y = rng.integers(0, 2, 2000)
s = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.2, 2000), 0, 1)
check("roc_auc matches sklearn", abs(M.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-9)

yt = [0, 0, 1, 1, 0, 1]
ys = [0.1, 0.4, 0.4, 0.8, 0.4, 0.9]      # ties present
check("roc_auc ties", abs(M.roc_auc(yt, ys) - roc_auc_score(yt, ys)) < 1e-9)

thr = M.threshold_at_fpr(y, s, 0.01)
realised = (s[y == 0] >= thr).mean()
check("threshold_at_fpr honours budget", realised <= 0.01 + 1e-9, f"got {realised}")

t = M.tpr_at_fpr(y, s, 0.01)
check("tpr_at_fpr in range", 0.0 <= t <= 1.0)
check("tpr monotone in fpr", M.tpr_at_fpr(y, s, 0.05) >= M.tpr_at_fpr(y, s, 0.01) - 1e-12)

pa = M.partial_auc(y, s, 0.1)
check("partial_auc standardised", 0.4 <= pa <= 1.0, f"got {pa}")

e, et = M.eer(y, s)
check("eer sane", 0.0 <= e <= 0.5, f"got {e}")
# perfect separation -> EER 0
py = np.array([0]*50 + [1]*50); ps = np.array([0.1]*50 + [0.9]*50)
check("eer perfect==0", M.eer(py, ps)[0] < 1e-9)

check("ece perfect calibration ~0", M.ece(np.array([0,1]*500), np.array([0.5]*1000)) < 0.05)
check("brier bounds", 0 <= M.brier(y, np.clip(s,0,1)) <= 1)

prf = M.precision_recall_f1(y, s, 0.6)
check("prf consistent", abs(prf["tp"]+prf["fp"]+prf["fn"]+prf["tn"] - len(y)) < 1e-9)

pt, lo, hi = M.bootstrap_ci(y, s, n=200)
check("bootstrap CI brackets point", lo <= pt <= hi, f"{lo} {pt} {hi}")

d, dlo, dhi = M.paired_bootstrap_delta(y, s, s, n=100)
check("paired delta of identical == 0", abs(d) < 1e-12 and abs(dlo) < 1e-9 and abs(dhi) < 1e-9)

summ = M.summary(y, s)
check("summary has headline keys", "tpr@fpr=0.01" in summ and "auc" in summ)


# ---------------------------------------------------------------- calibrate
check("sigmoid(logit(p))==p", np.allclose(C.sigmoid(C.logit([0.1,0.5,0.9])), [0.1,0.5,0.9], atol=1e-6))
check("logit clamps 0 and 1", np.isfinite(C.logit([0.0, 1.0])).all())

raw = np.clip(C.sigmoid(3.0 * C.logit(np.clip(s, 1e-3, 1-1e-3))), 1e-6, 1-1e-6)  # overconfident
ts = C.TemperatureScaler().fit(y, raw)
cal = ts.transform(raw)
check("temperature improves ECE", M.ece(y, cal) < M.ece(y, raw),
      f"{M.ece(y, cal):.4f} vs {M.ece(y, raw):.4f}")
check("temperature preserves AUC", abs(M.roc_auc(y, cal) - M.roc_auc(y, raw)) < 1e-9)

pl = C.PlattScaler().fit(y, raw); check("platt runs", np.isfinite(pl.transform(raw)).all())
iso = C.IsotonicCalibrator().fit(y, raw); check("isotonic runs", np.isfinite(iso.transform(raw)).all())

segs = np.array(["q_high" if v > 0.5 else "q_low" for v in rng.random(len(y))], dtype=object)
sc = C.SegmentedCalibrator(min_n=100).fit(y, raw, segs)
check("segmented fits >0 segments", sc.coverage()["segments_fitted"] >= 1)
check("segmented transform shape", sc.transform(raw, segs).shape == raw.shape)

# Gaussian score model: fit on synthetic two-cluster data with KNOWN generating
# parameters and require recovery -- not just "runs without error".
_gm_rng = np.random.default_rng(5)
_z0 = _gm_rng.normal(-2.0, 0.7, 4000)
_z1 = _gm_rng.normal(3.0, 1.2, 4000)
gm = C.fit_gaussian_score_model(C.sigmoid(_z0), C.sigmoid(_z1))
check("gaussian recovers class means",
      abs(gm.mu0 - (-2.0)) < 0.05 and abs(gm.mu1 - 3.0) < 0.05,
      f"mu0={gm.mu0:.3f} mu1={gm.mu1:.3f}")
check("gaussian recovers class stds",
      abs(gm.s0 - 0.7) < 0.05 and abs(gm.s1 - 1.2) < 0.05,
      f"s0={gm.s0:.3f} s1={gm.s1:.3f}")
check("posterior extreme at class centres",
      gm.posterior(C.sigmoid(np.array([-2.0])))[0] < 0.01
      and gm.posterior(C.sigmoid(np.array([3.0])))[0] > 0.99)
_t0, _t1 = gm.typicality(C.sigmoid(np.array([30.0])))   # far outside both
check("far sample atypical for both", _t0[0] > 3 and _t1[0] > 3)
check("open-set sample abstains",
      gm.bands(C.sigmoid(np.array([30.0])))[0] == "ABSTAIN")
_bnd = gm.decision_boundaries()
check("decision boundary between the means",
      any(-2.0 < b < 3.0 for b in _bnd), str(_bnd))


# --------------------------------------------------------------------- data
rows = []
for i in range(400):
    gen = ["authentic", "stylegan2", "sdxl", "faceswap-df", "sora-like"][i % 5]
    lab = 0 if gen == "authentic" else 1
    fam = {"authentic": "authentic", "stylegan2": "gan", "sdxl": "diffusion",
           "faceswap-df": "faceswap", "sora-like": "diffusion"}[gen]
    rows.append(dict(path=f"/d/{i}.jpg", label=lab, generator=gen, family=fam,
                     source_id=f"src{i % 40}", media_type="image", quality="native"))
man = pd.DataFrame(rows)
D.validate_manifest(man)
check("validate_manifest passes", True)

try:
    bad = man.copy(); bad.loc[0, "generator"] = "authentic"; bad.loc[0, "label"] = 1
    D.validate_manifest(bad); check("label-flip caught", False)
except ValueError:
    check("label-flip caught", True)

split = D.group_split(man, D.SplitSpec(holdout_generators=["sora-like"], seed=1))
check("all rows assigned", split["split"].isin(["train", "val", "test"]).all())
check("holdout gen only in test",
      set(split.loc[split.generator == "sora-like", "split"]) == {"test"})
rep = D.leakage_report(split, strict=False)
check("no shared source train|test", rep["train|test:shared_source_ids"] == 0, str(rep))
check("no shared generators train|test", rep["train|test:shared_generators"] == [], str(rep))
check("leakage_report clean", rep["clean"], str(rep))
check("balance_report runs", len(D.balance_report(split)) > 0)

img = (rng.random((32, 32)) * 255)
h1 = D.phash64(img)
h2 = D.phash64(img + rng.normal(0, 0.4, (32, 32)))
check("phash 64-bit", 0 <= h1 < 2**64)
check("phash stable under tiny noise", D.hamming64(h1, h2) <= 12, f"dist={D.hamming64(h1,h2)}")
hs = {"a": h1, "b": h1 ^ 0b1, "c": ~h1 & ((1 << 64) - 1)}
pairs = D.dedup_by_phash(hs, max_dist=6)
check("dedup finds the near-dupe", pairs and pairs[0][0] == ("a", "b"), str(pairs))


# ------------------------------------------------------------------ degrade
base = Image.fromarray((rng.random((128, 128, 3)) * 255).astype("uint8"))
eng = G.DegradationEngine(seed=3)
out, recipe = eng(base, severity="heavy")
check("degrade preserves size", out.size == base.size)
check("recipe recorded", "severity=heavy" in recipe and len(recipe.split("|")) >= 2, recipe)
check("jpeg changes pixels",
      not np.array_equal(np.asarray(G.jpeg(base, 25)), np.asarray(base)))
sweep = eng.sweep(base)
check("sweep returns all qualities", len(sweep) == 8 and all(v.size == base.size for v in sweep.values()))
check("screenshot_sim runs", G.screenshot_sim(base, np.random.default_rng(0)).size[0] > 0)
check("quality_bucket", G.quality_bucket(95) == "q_high" and G.quality_bucket(30) == "q_low"
      and G.quality_bucket(None) == "native")
# determinism given the same seed
a1, r1 = G.DegradationEngine(seed=7)(base)
a2, r2 = G.DegradationEngine(seed=7)(base)
check("degradation reproducible from seed", r1 == r2 and np.array_equal(np.asarray(a1), np.asarray(a2)))


# ------------------------------------------------------- audio degradation
from pk import audio_degrade as AD

sr_a = 16000
t_a = np.arange(sr_a * 2) / sr_a
# tone mix: one in-band (1 kHz), one above the telephone band (5 kHz)
wave = (0.3 * np.sin(2 * np.pi * 1000 * t_a)
        + 0.3 * np.sin(2 * np.pi * 5000 * t_a)).astype(np.float32)


def band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr_a)
    return float(spec[(f >= lo) & (f < hi)].sum())


g = AD.mu_law_roundtrip(wave, sr_a)
check("g711 preserves length", len(g) == len(wave))
check("g711 kills >4kHz (8k resample)",
      band_energy(g, 4500, 8000) < 0.01 * band_energy(wave, 4500, 8000))
check("g711 keeps in-band signal",
      band_energy(g, 900, 1100) > 0.5 * band_energy(wave, 900, 1100))

bp = AD.bandpass_telephone(wave, sr_a)
check("bandpass kills out-of-band",
      band_energy(bp, 4500, 8000) < 0.01 * band_energy(wave, 4500, 8000))

pl = AD.packet_loss(wave, sr_a, loss_rate=0.10, burst_ms=20,
                    rng=np.random.default_rng(3))
zeroed = float(np.mean(pl == 0.0))
check("packet loss drops ~rate", 0.02 < zeroed < 0.25, f"zeroed={zeroed}")
check("packet loss preserves length", len(pl) == len(wave))

rir = AD.synthetic_rir(sr_a, rt60=0.3, rng=np.random.default_rng(4))
rv = AD.convolve_rir(wave, rir)
check("rir preserves length", len(rv) == len(wave))
check("rir preserves level", abs(np.sqrt(np.mean(rv**2)) / np.sqrt(np.mean(wave**2)) - 1) < 0.05)

c1, rec1 = AD.TelephoneChannel(seed=11)(wave, sr_a)
c2, rec2 = AD.TelephoneChannel(seed=11)(wave, sr_a)
check("telephone channel reproducible from seed",
      rec1 == rec2 and np.array_equal(c1, c2))
check("telephone recipe recorded", "bandpass(300-3400)" in rec1)


# ------------------------------------------------------------------- fusion
n = 3000
truth = rng.integers(0, 2, n)
S = np.zeros((n, 6)); A = np.ones((n, 6))
strength = [1.6, 1.0, 1.2, 0.8, 0.4, 0.7]
for j in range(6):
    S[:, j] = np.clip(C.sigmoid(strength[j] * (truth * 2 - 1) + rng.normal(0, 1.0, n)), 1e-4, 1-1e-4)
A[rng.random(n) < 0.35, 4] = 0.0          # EXIF often STRIPPED

head = FU.FusionHead(seed=0).fit(S, A, truth, epochs=400, lr=0.3)
head.fit_temperature(S, A, truth)
p = head.predict_proba(S, A)
auc_f = M.roc_auc(truth, p)
best_single = max(M.roc_auc(truth, S[:, j]) for j in range(6))
check("fusion beats best single dimension", auc_f > best_single,
      f"fused={auc_f:.4f} best_single={best_single:.4f}")
check("fusion output is a probability", (p >= 0).all() and (p <= 1).all())

ev = FU.Evidence(
    scores={"ai_model": 0.93, "spectral": 0.88, "diffusion": 0.91,
            "temporal": 0.60, "exif": 0.50, "web_intel": 0.70},
    available={"ai_model": True, "spectral": True, "diffusion": True,
               "temporal": True, "exif": False, "web_intel": True})
res = head.score_one(ev)
check("score_one has API shape",
      {"confidence", "verdict", "fusion", "evidence", "driver"} <= set(res))
check("six evidence cards returned", len(res["evidence"]) == 6)
check("missing exif -> STRIPPED",
      [c for c in res["evidence"] if c["dimension"] == "exif"][0]["verdict"] == "STRIPPED")
check("missing card has null score",
      [c for c in res["evidence"] if c["dimension"] == "exif"][0]["score"] is None)
check("cards sorted by |contribution|",
      all(abs(res["evidence"][i]["contribution"]) >= abs(res["evidence"][i+1]["contribution"]) - 1e-12
          for i in range(5)))
check("verdict thresholds", FU.verdict_from_prob(0.9) == "SYNTHETIC"
      and FU.verdict_from_prob(0.6) == "SUSPICIOUS"
      and FU.verdict_from_prob(0.2) == "AUTHENTIC")
ab = head.abstention_rate(S, A)
check("abstention rate in [0,1]", 0 <= ab <= 1, f"{ab}")
dis = head.disagreement(S, A)
check("disagreement per row", dis.shape == (n,) and (dis >= 0).all())
# a row where all cards agree must have lower disagreement than a split row
agree = np.array([[0.9]*6]); split_row = np.array([[0.95, 0.05, 0.9, 0.1, 0.9, 0.1]])
ones = np.ones((1, 6))
check("disagreement discriminates",
      head.disagreement(agree, ones)[0] < head.disagreement(split_row, ones)[0])


# ------------------------------------------------------------------ harness
N = 4000
gen = rng.choice(["authentic", "stylegan2", "sdxl", "faceswap-df", "novel-gen"],
                 size=N, p=[0.5, 0.14, 0.14, 0.14, 0.08])
lab = (gen != "authentic").astype(int)
sep = {"authentic": 0.0, "stylegan2": 2.2, "sdxl": 1.9, "faceswap-df": 1.6, "novel-gen": 0.05}
sc = np.array([C.sigmoid(sep[g] + rng.normal(0, 1.0)) for g in gen])
deg = rng.choice(["none", "light", "heavy"], size=N, p=[0.5, 0.3, 0.2])
sc = np.where(deg == "heavy", np.clip(sc - 0.18 * lab, 1e-4, 1 - 1e-4), sc)
ef = pd.DataFrame(dict(label=lab, score=sc, generator=gen,
                       family=["authentic" if g == "authentic" else "synthetic" for g in gen],
                       quality=rng.choice(["q_high", "q_mid", "q_low"], size=N),
                       degradation=deg, prob=sc))

cfg = H.EvalConfig(bootstrap_n=100)
res1 = H.evaluate(ef, cfg, prob_col="prob")
check("harness overall computed", "tpr@fpr=0.01" in res1.overall)
check("per-generator excludes authentic",
      "authentic" not in [r["generator"] for r in res1.per_generator])
check("worst generator first",
      res1.per_generator[0]["tpr@fpr=0.01"] <= res1.per_generator[-1]["tpr@fpr=0.01"])
check("harness finds the blind spot", res1.per_generator[0]["generator"] == "novel-gen",
      str([r["generator"] for r in res1.per_generator]))
check("robustness curve present", len(res1.robustness) == 3)
check("heavy degradation hurts",
      min(r["auc"] for r in res1.robustness if r["degradation"] == "heavy")
      < max(r["auc"] for r in res1.robustness if r["degradation"] == "none"))
check("auc CI present", len(res1.overall["auc_ci95"]) == 2)

md = H.to_markdown(res1, cfg)
check("markdown report renders", "Per generator" in md and "Coverage gaps" in md)
check("coverage gap flagged", "novel-gen" in md)

# release gate: candidate that regresses one generator must be blocked
ef2 = ef.copy()
mask = ef2.generator == "faceswap-df"
ef2.loc[mask, "score"] = np.clip(ef2.loc[mask, "score"] - 0.35, 1e-4, 1 - 1e-4)
res2 = H.evaluate(ef2, cfg, prob_col="prob")
gate = H.release_gate(res2, res1, cfg)
check("gate blocks per-generator regression", gate["promote"] is False, str(gate["reasons"]))
gate_same = H.release_gate(res1, res1, cfg)
check("gate passes identical model", gate_same["promote"] is True, str(gate_same["reasons"]))

path = os.path.join(tempfile.gettempdir(), "_eval.json"); res1.to_json(path)
check("json export", os.path.exists(path) and os.path.getsize(path) > 100)


# ----------------------------------------------------------------- finetune
from pk import finetune as FT
buf = FT.RehearsalBuffer(capacity_per_generator=50, seed=0)
for g in ["stylegan2", "sdxl", "faceswap-df"]:
    buf.extend(g, [f"{g}-{i}" for i in range(500)])
check("reservoir respects capacity", all(v == 50 for v in buf.stats().values()), str(buf.stats()))
sample = buf.sample(30, exclude="sdxl")
check("sample excludes generator", all("sdxl" not in x for x in sample))
mixed = buf.mixed_batch([f"new-{i}" for i in range(32)], ratio=0.5)
n_new = sum(1 for x in mixed if x.startswith("new-"))
check("mixed batch ~50% replay", 0.35 <= n_new / len(mixed) <= 0.65, f"{n_new}/{len(mixed)}")
check("cosine schedule warmup", FT.cosine_schedule(0, 100) == 0.0
      and 0.9 < FT.cosine_schedule(10, 100) <= 1.0
      and FT.cosine_schedule(99, 100) < 0.05)

if FT.TORCH:
    import torch, torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({"q_proj": nn.Linear(32, 32),
                               "v_proj": nn.Linear(32, 32),
                               "mlp": nn.Linear(32, 32),
                               "ln": nn.LayerNorm(32)}) for _ in range(4)])
            self.head = nn.Linear(32, 2)

        def forward(self, x):
            for b in self.blocks:
                x = b["ln"](x + b["mlp"](b["q_proj"](x) + b["v_proj"](x)))
            return self.head(x)

    m = Tiny()
    tr, tot, ratio = FT.freeze_all_but_layernorm(m)
    check("LN-tuning trains few params", ratio < 0.05, f"ratio={ratio:.4f}")
    check("LN params unfrozen", tr == 4 * 2 * 32 + 32 * 2 + 2, f"tr={tr}")
    check("trainable_report renders", "trainable" in FT.trainable_report(m))

    m2 = Tiny()
    x = torch.randn(8, 32)
    before = m2(x).detach().clone()
    k = FT.inject_lora(m2, r=4, alpha=8)
    check("lora injected into q and v", k == 8, f"replaced={k}")
    after = m2(x).detach()
    check("lora is a no-op at init (B=0)", torch.allclose(before, after, atol=1e-6))
    tr2, tot2 = FT.count_params(m2)
    check("only lora params trainable", 0 < tr2 < tot2 * 0.2, f"{tr2}/{tot2}")
    # one optimisation step must change the output
    opt = torch.optim.Adam([p for p in m2.parameters() if p.requires_grad], lr=0.1)
    loss = m2(x).pow(2).mean(); loss.backward(); opt.step()
    check("lora params receive gradient", not torch.allclose(before, m2(x).detach(), atol=1e-6))

    hs = FT.HypersphereHead(32)
    logits = hs(torch.randn(5, 32))
    check("hypersphere head shape", tuple(logits.shape) == (5, 2))
    check("cosine logits bounded by scale",
          logits.abs().max().item() <= hs.scale.item() + 1e-4)
else:
    PASS.append("torch tests skipped (torch not installed)")


print(f"\n{'='*64}")
print(f"  ALL {len(PASS)} CHECKS PASSED")
print(f"{'='*64}")
for p in PASS:
    print(f"  ok  {p}")
