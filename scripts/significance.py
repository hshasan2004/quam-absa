"""Paired bootstrap significance test: QuAM-xLSTM vs. a fine-tuned baseline on the same test reviews.

    python scripts/significance.py --split official \
        --ours hyb_official hyb_official_s1 hyb_official_s2 \
        --base bl_banglabert_official bl_banglabert_official_s1 bl_banglabert_official_s2

Statistic: strict pair F1 (micro), averaged over the seeds of each system. Test reviews are
resampled with replacement (paired: the same sample for both systems); p = fraction of bootstrap
samples in which ours does not beat the baseline. Each run uses its own validation-chosen
threshold. Our runs are re-evaluated from best.pt; baseline predictions come from test_preds.npz.
Writes reports/significance_<split>.json.
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
from train import make_loader  # noqa: E402


def per_review_counts(A, threshold, review_ids):
    """(tp, n_pred, n_gold) per review for strict pair matching, ordered like review_ids."""
    pred = A["prob"] > threshold
    gold = A["gold_pres"] == 1
    tp = pred & gold & (A["pred_sent"] == A["gold_sent"])
    idx = {r: i for i, r in enumerate(review_ids)}
    out = np.zeros((len(review_ids), 3))
    rows = np.array([idx[r] for r in A["id"]])
    np.add.at(out[:, 0], rows, tp)
    np.add.at(out[:, 1], rows, pred)
    np.add.at(out[:, 2], rows, gold)
    return out


def f1(c):
    p = c[..., 0] / np.maximum(c[..., 1], 1)
    r = c[..., 0] / np.maximum(c[..., 2], 1)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def ours_preds(names):
    cache = EncoderCache(default_cache_dir())
    out = []
    for n in names:
        run, cfg, met = load_run(n)
        recs = load_records(("eco",))
        test_ds, _, model = build(cfg, cache, recs)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        A, _, _ = collect(model.to("cuda"), make_loader(test_ds, 64, False, 0), "cuda")
        out.append((A, met["test"]["eco"]["threshold"]))
        del model; torch.cuda.empty_cache()
    return out


def base_preds(names):
    out = []
    for n in names:
        z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
        A = {k[2:]: z[k] for k in z.files if k.startswith("A_")}
        out.append((A, float(z["threshold"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    O, B = ours_preds(args.ours), base_preds(args.base)
    ids = sorted(set(O[0][0]["id"]))
    for A, _ in O + B:                                       # same test reviews and aspect rows
        assert sorted(set(A["id"])) == ids, "systems were evaluated on different test reviews"
    co = np.stack([per_review_counts(A, t, ids) for A, t in O])   # (seeds, reviews, 3)
    cb = np.stack([per_review_counts(A, t, ids) for A, t in B])

    full_o, full_b = f1(co.sum(1)).mean(), f1(cb.sum(1)).mean()
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n)
    for i in range(args.n):
        s = rng.integers(0, len(ids), len(ids))
        diffs[i] = f1(co[:, s].sum(1)).mean() - f1(cb[:, s].sum(1)).mean()
    res = {"split": args.split, "ours": args.ours, "baseline": args.base, "n_reviews": len(ids),
           "ours_pairF1": 100 * full_o, "baseline_pairF1": 100 * full_b, "diff": 100 * (full_o - full_b),
           "diff_95CI": [100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)],
           "p_value": float((diffs <= 0).mean()), "bootstrap_samples": args.n,
           "per_seed_ours": [100 * f for f in f1(co.sum(1))], "per_seed_base": [100 * f for f in f1(cb.sum(1))]}
    out = ROOT / "reports" / f"significance_{args.split}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
