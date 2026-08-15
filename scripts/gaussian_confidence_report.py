#!/usr/bin/env python3
"""
Gaussian-motivated confidence for the audio track: fit class-conditional
Gaussians on TRAIN probe scores (authentic vs Kokoro, logit space), then read
posterior + LLR + typicality for every held-out test clip.

The scenario logic, stated honestly: the mentor's question is "what does the
score mean when an engine you never trained on shows up (ElevenLabs, Fish
Audio)". We have no clips from either engine, so nothing here claims to
validate them. What we do have is Piper -- a synthesizer genuinely unseen at
train time -- as the closest real stand-in, plus the telephone-degraded copy
of the same clips as a second scenario. A real ElevenLabs/Fish Audio check
needs actual samples from those engines: future work, not something to fake.

Outputs:
  data/gaussian_confidence_test.csv   per-clip posterior/LLR/typicality/band
  data/gaussian_confidence.png        fitted Gaussians + test scores, one image
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pk import calibrate as C  # noqa: E402

TAP = 6
CACHE = Path("data/features/w2v2_layers.npz")
SCORES_CSV = Path("data/degraded_scores.csv")
OUT_CSV = Path("data/gaussian_confidence_test.csv")
OUT_PNG = Path("data/gaussian_confidence.png")
ABSTAIN_SIGMA = 3.0


def key_for(p, L=TAP):
    return f"{Path(p).as_posix()}::L{L}"


def main() -> None:
    df = pd.read_csv("data/manifest.csv")
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    with np.load(CACHE) as z:
        feats = {k: z[k] for k in z.files}

    # ---- probe, identical recipe to probe_layers/eval_degraded (deterministic)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = df[df.split == "train"]
    Xtr = np.stack([feats[key_for(p)] for p in train["path"]])
    ytr = train["label"].to_numpy()
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    probe.fit(Xtr, ytr)

    # ---- fit the two Gaussians on TRAIN scores (authentic vs kokoro)
    p_tr = probe.predict_proba(Xtr)[:, 1]
    gm = C.fit_gaussian_score_model(p_tr[ytr == 0], p_tr[ytr == 1])
    print("fitted Gaussians in logit space (TRAIN: authentic vs kokoro):")
    print(f"  authentic: mu={gm.mu0:+.2f}  sigma={gm.s0:.2f}")
    print(f"  spoof    : mu={gm.mu1:+.2f}  sigma={gm.s1:.2f}")
    print(f"  decision boundary z (LLR=0, equal priors): "
          f"{['%.2f' % b for b in gm.decision_boundaries()]}")
    print(f"  NOTE: fitted on in-sample train scores per plan; in-sample margins")
    print(f"  are optimistically wide, so treat sigmas as lower bounds.")

    # in-sample fit honesty check on val (held out, same generator family)
    val = df[df.split == "val"]
    Xv = np.stack([feats[key_for(p)] for p in val["path"]])
    pv = probe.predict_proba(Xv)[:, 1]
    t0v, t1v = gm.typicality(pv)
    print(f"  val check: median typicality of val-authentic from auth Gaussian "
          f"{np.median(t0v[val.label.to_numpy() == 0]):.1f} sigma; "
          f"val-kokoro from spoof Gaussian "
          f"{np.median(t1v[val.label.to_numpy() == 1]):.1f} sigma\n")

    # ---- test scenarios from the stored eval scores
    dd = pd.read_csv(SCORES_CSV)
    # consistency guard: the CSV's frozen scores must match this probe exactly
    clean = dd[dd.degradation == "none"].reset_index(drop=True)
    Xc = np.stack([feats[key_for(p)] for p in clean["path"]])
    assert np.allclose(probe.decision_function(Xc), clean["score_frozen"], atol=1e-6), \
        "stored scores do not match the refit probe -- pipelines diverged"

    rows = []
    for scen, sub in [("clean", clean),
                      ("telephone", dd[dd.degradation == "telephone"].reset_index(drop=True))]:
        p = C.sigmoid(sub["score_frozen"].to_numpy())   # logit(sigmoid(z)) == z
        post = gm.posterior(p)
        llr = gm.llr(p)
        t0, t1 = gm.typicality(p)
        band = gm.bands(p, abstain_sigma=ABSTAIN_SIGMA)
        for i in range(len(sub)):
            r = sub.iloc[i]
            rows.append(dict(scenario=scen, path=r["path"], label=int(r["label"]),
                             generator=r["generator"],
                             posterior=float(post[i]), llr=float(llr[i]),
                             sigmas_from_authentic=float(t0[i]),
                             sigmas_from_spoof=float(t1[i]), band=band[i]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"per-clip confidence -> {OUT_CSV} ({len(out)} rows)")

    # ---- the unseen-synthesizer question, answered plainly
    for scen in ("clean", "telephone"):
        s = out[out.scenario == scen]
        print(f"\n=== scenario: {scen} ===")
        for gen in ("authentic", "piper"):
            g = s[s.generator == gen]
            t0, t1 = g["sigmas_from_authentic"], g["sigmas_from_spoof"]
            print(f"  {gen:10s} n={len(g):2d}  "
                  f"sig_from_auth med={t0.median():5.1f} [{t0.min():.1f},{t0.max():5.1f}]  "
                  f"sig_from_spoof med={t1.median():5.1f} [{t1.min():.1f},{t1.max():5.1f}]  "
                  f"posterior med={g.posterior.median():.3f}")
        print("  bands: " + "  ".join(
            f"{gen}: {dict(s[s.generator == gen].band.value_counts())}"
            for gen in ("authentic", "piper")))

    piper_clean = out[(out.scenario == "clean") & (out.generator == "piper")]
    inside = float((piper_clean.sigmas_from_spoof <= ABSTAIN_SIGMA).mean())
    outside_both = float(((piper_clean.sigmas_from_spoof > ABSTAIN_SIGMA)
                          & (piper_clean.sigmas_from_authentic > ABSTAIN_SIGMA)).mean())
    print(f"\nUNSEEN-SYNTHESIZER READ (piper, clean): "
          f"{inside:.0%} of clips sit within {ABSTAIN_SIGMA:.0f} sigma of the fitted "
          f"spoof Gaussian; {outside_both:.0%} sit outside BOTH (open-set zone).")
    print("Piper is the stand-in for the mentor's ElevenLabs / Fish Audio scenario:")
    print("a genuine validation for those engines requires actual samples from them")
    print("-- future work, deliberately not simulated here.")

    # ---- the one-image explanation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_auth = C.logit(C.sigmoid(clean.loc[clean.label == 0, "score_frozen"]), eps=1e-12)
    z_pip = C.logit(C.sigmoid(clean.loc[clean.label == 1, "score_frozen"]), eps=1e-12)
    zz = np.linspace(min(gm.mu0 - 4 * gm.s0, float(np.min(z_auth)) - 1),
                     max(gm.mu1 + 4 * gm.s1, float(np.max(z_pip)) + 1), 600)

    def pdf(z, mu, s):
        return np.exp(-0.5 * ((z - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(zz, pdf(zz, gm.mu0, gm.s0), color="tab:blue",
            label=f"fitted authentic N({gm.mu0:.1f}, {gm.s0:.1f}$^2$)")
    ax.plot(zz, pdf(zz, gm.mu1, gm.s1), color="tab:red",
            label=f"fitted spoof/kokoro N({gm.mu1:.1f}, {gm.s1:.1f}$^2$)")
    for lo, hi, col in [(gm.mu0 - 3 * gm.s0, gm.mu0 + 3 * gm.s0, "tab:blue"),
                        (gm.mu1 - 3 * gm.s1, gm.mu1 + 3 * gm.s1, "tab:red")]:
        ax.axvspan(lo, hi, color=col, alpha=0.06)
    ax.hist(z_auth, bins=15, density=True, alpha=0.45, color="tab:blue",
            label="test authentic (held-out)")
    ax.hist(z_pip, bins=15, density=True, alpha=0.45, color="tab:red",
            label="test piper (UNSEEN synthesizer)")
    ax.plot(z_auth, np.full_like(z_auth, -0.004), "|", color="tab:blue", ms=12)
    ax.plot(z_pip, np.full_like(z_pip, -0.010), "|", color="tab:red", ms=12)
    for b in gm.decision_boundaries():
        ax.axvline(b, color="gray", ls="--", lw=1,
                   label=f"LLR=0 at z={b:.1f}")
    ax.set_xlabel("z = logit(probe score)")
    ax.set_ylabel("density")
    ax.set_title("Class-conditional Gaussians (fit: train auth vs kokoro) "
                 "with held-out test scores")
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nfigure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
