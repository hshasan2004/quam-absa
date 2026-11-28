"""Seed ensemble of saved test predictions (no retraining, no test-set tuning).

    python scripts/ensemble_eval.py            -> reports/ensemble.json
Presence probability = mean over seeds; threshold = mean of each seed's validation-tuned threshold;
polarity = majority vote over seeds (ties -> first seed). Reports pair F1, detection macro-F1 and,
on the zero-shot split, pair / detection F1 on unseen aspect names.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

recs = load_records(("eco",))
seen = {a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}
GROUPS = {"official": {"no retrieval": ["ftp_nomem_e15_official", "ftp_nomem_e15_official_s1", "ftp_nomem_e15_official_s2"],
                       "rag3": ["rag3_official", "rag3_official_s1", "rag3_official_s2"],
                       "hand list": ["kb_official_c", "kb_official_s1", "kb_official_s2"]},
          "zeroshot": {"no retrieval": ["ftp_nomem_e15_zeroshot", "ftp_nomem_e15_zeroshot_s1", "ftp_nomem_e15_zeroshot_s2"],
                       "rag3": ["rag3_zeroshot", "rag3_zeroshot_s1", "rag3_zeroshot_s2"],
                       "hand list": ["kb_zeroshot", "kb_zeroshot_s1", "kb_zeroshot_s2"]}}


def f1(tp, p, g):
    return 200 * tp / max(p + g, 1)


def score(name, pred, sent, gold, gsent, unseen):
    ok = pred & gold & (sent == gsent)
    out = {"pair_F1": f1(ok.sum(), pred.sum(), gold.sum())}
    mf = [f1((pred & gold & (name == n)).sum(), (pred & (name == n)).sum(), (gold & (name == n)).sum())
          for n in np.unique(name[gold])]
    out["det_macroF1"] = float(np.mean(mf))
    if unseen is not None:
        u = unseen
        out["unseen_pair_F1"] = f1((ok & u).sum(), (pred & u).sum(), (gold & u).sum())
        out["unseen_det_F1"] = f1((pred & gold & u).sum(), (pred & u).sum(), (gold & u).sum())
    return out


def load(n):
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    return {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])


def ensemble(runs, prob_fn=None):
    """prob_fn(run_name) may return replacement (A, threshold) for a run, e.g. re-scored predictions."""
    P = [prob_fn(n) if prob_fn else load(n) for n in runs]
    A0 = P[0][0]
    for A, _ in P[1:]:
        assert (A["id"] == A0["id"]).all() and (A["name"] == A0["name"]).all(), "row order differs"
    prob = np.mean([A["prob"] for A, _ in P], 0); t = float(np.mean([t for _, t in P]))
    S = np.stack([A["pred_sent"] for A, _ in P])
    vote = np.array([np.bincount(c, minlength=3).argmax() if np.bincount(c, minlength=3).max() > 1 else c[0] for c in S.T])
    return A0, prob > t, vote


if __name__ == "__main__":
    res = {}
    for split, groups in GROUPS.items():
        for g, runs in groups.items():
            A0, pred, vote = ensemble(runs)
            unseen = np.array([x not in seen for x in A0["name"]]) if split == "zeroshot" else None
            gold = A0["gold_pres"] == 1
            single = []
            for n in runs:
                A, t = load(n)
                single.append(score(A["name"], A["prob"] > t, A["pred_sent"], gold, A["gold_sent"], unseen))
            ens = score(A0["name"], pred, vote, gold, A0["gold_sent"], unseen)
            mean = {k: float(np.mean([s[k] for s in single])) for k in ens}
            res[f"{split}|{g}"] = {"seed_mean": mean, "ensemble": ens}
            print(f"{split:8s} {g:13s} " + "  ".join(f"{k} {mean[k]:5.1f}->{ens[k]:5.1f}" for k in ens))
    (ROOT / "reports" / "ensemble.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
