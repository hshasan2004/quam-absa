"""Paired bootstrap on BABSA: aspect sentiment macro-F1 (gold aspects), ours vs a baseline.

    python scripts/significance_babsa.py --ours A B C --base X Y Z --name banglabert
Each system gives a 3x3 confusion matrix per review (same gold aspects per review); reviews are resampled with
replacement; macro-F1 is averaged over the seeds of each system. Writes reports/sig_gdn/babsa_vs_<name>.json.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(n):
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    keep = z["A_gold_pres"] == 1
    return z["A_id"][keep], z["A_gold_sent"][keep], z["A_pred_sent"][keep]


def confusions(n, reviews):
    ids, g, p = load(n)
    idx = {r: k for k, r in enumerate(reviews)}
    C = np.zeros((len(reviews), 3, 3))
    np.add.at(C, (np.array([idx[r] for r in ids]), g, p), 1)
    return C


def macro_f1(C):                            # C (..., 3, 3): gold x pred
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    return np.mean(2 * tp / np.maximum(C.sum(-1) + C.sum(-2), 1), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True); ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    reviews = sorted(set(load(a.ours[0])[0])); no = len(a.ours)
    C = np.stack([confusions(n, reviews) for n in a.ours + a.base])      # (systems, reviews, 3, 3)
    assert (C.sum(-1) == C[0].sum(-1)).all(), "systems have different gold aspects"
    full = macro_f1(C.sum(1)); rng = np.random.default_rng(0); diffs = np.empty(a.n)
    for b in range(a.n):
        w = np.bincount(rng.integers(0, len(reviews), len(reviews)), minlength=len(reviews))
        m = macro_f1(np.einsum("r,srij->sij", w, C)); diffs[b] = m[:no].mean() - m[no:].mean()
    rev, ids = reviews, C[0].sum((1, 2))
    res = {"ours": a.ours, "baseline": a.base, "n_reviews": len(rev), "n_aspects": int(ids.sum()),
           "ours_macroF1": 100 * full[:no].mean(), "baseline_macroF1": 100 * full[no:].mean(),
           "diff": 100 * (full[:no].mean() - full[no:].mean()),
           "diff_95CI": [100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)],
           "p_value": float((diffs <= 0).mean()), "per_seed_ours": list(100 * full[:no]),
           "per_seed_base": list(100 * full[no:])}
    out = ROOT / "reports" / "sig_gdn" / f"babsa_vs_{a.name}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8"); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
