"""Zero-shot split: detection / pair F1 on aspect names seen vs. never seen in training.

    python scripts/unseen_aspects.py --device cpu

Systems (all seeds that exist): QuAM v2, its no-memory variant, and the baselines (from their
test_preds.npz). Writes reports/unseen_aspects.json (mean/std over seeds).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze import build, cache_for, collect, load_run  # noqa: E402
from quam.data import load_records  # noqa: E402
from train import make_loader  # noqa: E402

OURS = ["q2_zeroshot", "q2only_zeroshot"]
BASE = ["bl_xlmr_frozen_zeroshot", "bl_banglabert_zeroshot", "bl_xlmr_zeroshot",
        "ftp_zeroshot", "ft_zeroshot"]          # last two: QuAM-FT (from test_preds.npz)


def f1s(A, t, mask):
    pred, gold = (A["prob"] > t) & mask, (A["gold_pres"] == 1) & mask
    f = lambda tp: 200 * tp / max(pred.sum() + gold.sum(), 1)
    return {"det_F1": f((pred & gold).sum()), "pair_F1": f((pred & gold & (A["pred_sent"] == A["gold_sent"])).sum()),
            "n_gold": int(gold.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    dev = ap.parse_args().device
    recs = load_records(("eco",))
    seen = {a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}
    out = {}
    for base in OURS + BASE:
        per_seed = []
        for n in (base, f"{base}_s1", f"{base}_s2"):
            if not (ROOT / "runs" / n / "metrics.json").exists():
                continue
            if base in OURS:
                run, cfg, met = load_run(n)
                test_ds, _, model = build(cfg, cache_for(cfg), recs)
                model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
                A, _, _ = collect(model.to(dev), make_loader(test_ds, 64, False, 0), dev)
                t = met["test"]["eco"]["threshold"]
                del model
            else:
                z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
                A, t = {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])
            uns = np.array([a not in seen for a in A["name"]])
            per_seed.append({"seen": f1s(A, t, ~uns), "unseen": f1s(A, t, uns)})
        if not per_seed:
            continue
        agg = {"n_seeds": len(per_seed)}
        for part in ("seen", "unseen"):
            for k in ("det_F1", "pair_F1"):
                v = [s[part][k] for s in per_seed]
                agg[f"{part}_{k}"] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0}
            agg[f"{part}_n_gold"] = per_seed[0][part]["n_gold"]
        out[base] = agg
        print(base, {k: (f"{v['mean']:.1f}±{v['std']:.1f}" if isinstance(v, dict) else v) for k, v in agg.items()})
    (ROOT / "reports" / "unseen_aspects.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
