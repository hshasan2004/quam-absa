"""Paired bootstrap between memory variants of the aspect-prompted model (3 seeds each).

    python scripts/memory_significance.py   -> reports/memory_significance.json
Statistics: pair F1 (micro), detection macro-F1 over aspect names, and on the zero-shot split pair
and detection F1 on unseen aspect names. Reviews are resampled (paired); seeds are averaged.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

recs = load_records(("eco",))
seen = {a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}
SYS = {"gdn": "ftp_gdn", "xlstm": "ftp", "none": "ftp_nomem"}
COMP = [("gdn", "none"), ("gdn", "xlstm"), ("xlstm", "none")]


def counts(n, ids, names):
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    A, t = {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])
    pred, gold = A["prob"] > t, A["gold_pres"] == 1
    tp_p, tp_d = pred & gold & (A["pred_sent"] == A["gold_sent"]), pred & gold
    r = np.array([ids[i] for i in A["id"]]); a = np.array([names[x] for x in A["name"]])
    C = np.zeros((len(ids), len(names), 4))           # per review, per aspect name: tp_pair, tp_det, pred, gold
    for j, v in enumerate((tp_p, tp_d, pred, gold)):
        np.add.at(C[..., j], (r, a), v)
    return C


def f1(tp, p, g):
    return 2 * tp / np.maximum(p + g, 1)


def stats(C, unseen_mask):                            # C (reviews, names, 4) summed with weights
    S = C.sum(0)
    tot = S.sum(0)
    has = S[:, 3] > 0
    out = {"pair_F1": f1(tot[0], tot[2], tot[3]),
           "det_macroF1": f1(S[has, 1], S[has, 2], S[has, 3]).mean()}
    if unseen_mask is not None:
        u = S[unseen_mask].sum(0)
        out["unseen_pair_F1"], out["unseen_det_F1"] = f1(u[0], u[2], u[3]), f1(u[1], u[2], u[3])
    return out


res = {}
for split in ("official", "zeroshot"):
    runs = {k: [f"{b}_{split}" + s for s in ("", "_s1", "_s2")] for k, b in SYS.items()}
    z = np.load(ROOT / "runs" / runs["gdn"][0] / "test_preds.npz", allow_pickle=True)
    ids = {r: i for i, r in enumerate(sorted(set(z["A_id"])))}; names = {n: i for i, n in enumerate(sorted(set(z["A_name"])))}
    um = np.array([n not in seen for n in names]) if split == "zeroshot" else None
    C = {k: np.stack([counts(n, ids, names) for n in v]) for k, v in runs.items()}
    rng = np.random.default_rng(0); B = 2000
    W = [np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(B)]
    full = {k: [stats(c, um) for c in C[k]] for k in C}
    boot = {k: [] for k in C}
    for w in W:
        for k in C:
            boot[k].append([stats(c * w[:, None, None], um) for c in C[k]])
    for a, b in COMP:
        for m in full[a][0]:
            fa, fb = np.mean([s[m] for s in full[a]]), np.mean([s[m] for s in full[b]])
            d = np.array([np.mean([s[m] for s in x]) - np.mean([s[m] for s in y]) for x, y in zip(boot[a], boot[b])])
            res[f"{split}|{a}_vs_{b}|{m}"] = {"a": 100 * fa, "b": 100 * fb, "diff": 100 * (fa - fb),
                                              "CI": [100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)],
                                              "p": float((d <= 0).mean())}
            r = res[f"{split}|{a}_vs_{b}|{m}"]
            print(f"{split:8s} {a:5s} vs {b:5s} {m:15s} {r['a']:5.1f} {r['b']:5.1f} {r['diff']:+5.2f} [{r['CI'][0]:+.2f},{r['CI'][1]:+.2f}] p={r['p']:.3f}")
(ROOT / "reports" / "memory_significance.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
