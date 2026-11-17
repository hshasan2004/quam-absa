"""Seed ensemble: average the predicted probabilities of several trained runs.

    python scripts/ensemble.py --name ens_hybrid fin_hybrid fin_hybrid_s1 fin_hybrid_s2

All runs must use the same data and split. Presence probabilities, aspect-sentiment and
overall-sentiment distributions are averaged; the detection threshold is chosen on the
validation ensemble and applied to test. Writes runs/<name>/metrics.json (+ config.json),
so summarize.py lists it like any other run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import build, collect, load_run  # noqa: E402
from quam.data import EncoderCache, default_cache_dir, load_records  # noqa: E402
from quam.metrics import babsa_metrics, best_eco_threshold, eco_metrics  # noqa: E402
from train import make_loader  # noqa: E402


def average(parts):
    """parts: list of (A, R) with identical row order -> averaged (A, R) with argmax predictions."""
    A0, R0 = parts[0]
    for A, R in parts[1:]:
        assert (A["id"] == A0["id"]).all() and (A["name"] == A0["name"]).all() and (R["id"] == R0["id"]).all()
    A, R = dict(A0), dict(R0)
    A["prob"] = np.mean([p[0]["prob"] for p in parts], 0)
    A["sprob"] = np.mean([np.stack(p[0]["sprob"]) for p in parts], 0)
    A["pred_sent"] = A["sprob"].argmax(-1)
    R["oprob"] = np.mean([np.stack(p[1]["oprob"]) for p in parts], 0)
    R["pred"] = R["oprob"].argmax(-1)
    return A, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    cache = EncoderCache(default_cache_dir())
    val_parts, test_parts, cfg0 = [], [], None
    for name in args.runs:
        run, cfg, _ = load_run(name)
        if cfg0 is None:
            cfg0 = cfg
        assert (cfg["data"], cfg["eco_split"]) == (cfg0["data"], cfg0["eco_split"]), "runs use different data"
        which = {"eco": ("eco",), "babsa": ("babsa",), "both": ("eco", "babsa")}[cfg["data"]]
        recs = load_records(which)
        test_ds, val_ds, model = build(cfg, cache, recs, eval_seed=42)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        model.to("cuda")
        Av, Rv, _ = collect(model, make_loader(val_ds, 64, False, 0), "cuda")
        At, Rt, _ = collect(model, make_loader(test_ds, 64, False, 0), "cuda")
        val_parts.append((Av, Rv)); test_parts.append((At, Rt))
        del model; torch.cuda.empty_cache()
        print(f"  collected {name}", flush=True)

    (Av, Rv), (At, Rt) = average(val_parts), average(test_parts)
    val, test = {}, {}
    if cfg0["data"] in ("eco", "both"):
        t = best_eco_threshold(Av, Rv)
        val["eco"], test["eco"] = eco_metrics(Av, Rv, t), eco_metrics(At, Rt, t)
    if cfg0["data"] in ("babsa", "both"):
        val["babsa"], test["babsa"] = babsa_metrics(Av), babsa_metrics(At)
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg0, name=args.name, ensemble_of=args.runs, seed="ensemble")
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"best_epoch": "–", "val": val, "test": test}, indent=1,
                                                 default=float), encoding="utf-8")
    print("TEST " + json.dumps(test, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
