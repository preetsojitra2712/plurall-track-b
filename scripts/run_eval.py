#!/usr/bin/env python3
"""
Run the eval harness over a scored CSV and write a report.

    python scripts/run_eval.py --scores scored.csv --out reports/v1
    python scripts/run_eval.py --scores cand.csv --baseline reports/v0.json --out reports/v1

Expected columns in --scores:
    label, score, generator            (required)
    family, quality, degradation, prob (optional, all improve the report)

With --baseline, it also runs the release gate and exits non-zero if the
candidate would not be promoted. That makes it usable as a CI step:

    python scripts/run_eval.py --scores cand.csv --baseline reports/prod.json --out /tmp/r || exit 1
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd

from pk import harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV with label,score,generator,...")
    ap.add_argument("--out", default="reports/run", help="output prefix (no extension)")
    ap.add_argument("--baseline", default=None, help="incumbent eval JSON for the release gate")
    ap.add_argument("--primary-fpr", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--title", default="Eval report")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    cfg = H.EvalConfig(primary_fpr=args.primary_fpr, bootstrap_n=args.bootstrap)
    prob_col = "prob" if "prob" in df.columns else None
    res = H.evaluate(df, cfg, prob_col=prob_col)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(out.with_suffix(".json"))
    md = H.to_markdown(res, cfg, title=args.title)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        inc = H.EvalResult(**base)
        gate = H.release_gate(res, inc, cfg)
        print("\n" + "=" * 60)
        print(f"RELEASE GATE: {'PROMOTE' if gate['promote'] else 'BLOCKED'}")
        for r in gate["reasons"]:
            print(f"  - {r}")
        print("=" * 60)
        sys.exit(0 if gate["promote"] else 1)


if __name__ == "__main__":
    main()
