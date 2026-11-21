"""Where does the xLSTM memory help? Pair F1 of with-memory vs no-memory runs per review group.

    python scripts/memory_groups.py
Groups: script, review length (tokens, terciles), number of gold aspects, seen/unseen aspect name (zero-shot).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

PAIRS = [("ftp_official", "ftp_nomem_official"), ("ftp_zeroshot", "ftp_nomem_zeroshot"),
         ("ftp_f9_official", "ftp_nomem_f9_official"), ("ftp_f12_official", "ftp_nomem_f12_official")]


def load(n):
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    return {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])


def pf1(A, t, m):
    pred, gold = (A["prob"] > t) & m, (A["gold_pres"] == 1) & m
    tp = (pred & gold & (A["pred_sent"] == A["gold_sent"])).sum()
    return 200 * tp / max(pred.sum() + gold.sum(), 1), int(gold.sum())


recs = {r["id"]: r for r in load_records(("eco",))}
seen = {a["query"] for r in recs.values() if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}
for ours, nomem in PAIRS:
    if not all((ROOT / "runs" / n / "test_preds.npz").exists() for n in (ours, nomem)):
        continue
    (A, ta), (B, tb) = load(ours), load(nomem)
    assert (A["id"] == B["id"]).all() and (A["name"] == B["name"]).all()
    ids = A["id"]
    script = np.array([recs[i]["script"] for i in ids])
    length = np.array([len(recs[i]["text"].split()) for i in ids])
    ngold = {}
    for i, g in zip(ids, A["gold_pres"]):
        ngold[i] = ngold.get(i, 0) + int(g == 1)
    nga = np.array([ngold[i] for i in ids])
    q1, q2 = np.percentile([len(r["text"].split()) for r in recs.values()], [33, 66])
    groups = {"all": np.ones(len(ids), bool)}
    for s in np.unique(script):
        groups[f"script={s}"] = script == s
    groups[f"len<={q1:.0f}"] = length <= q1
    groups[f"len {q1:.0f}-{q2:.0f}"] = (length > q1) & (length <= q2)
    groups[f"len>{q2:.0f}"] = length > q2
    groups["gold aspects<=1"] = nga <= 1
    groups["gold aspects=2"] = nga == 2
    groups["gold aspects>=3"] = nga >= 3
    if "zeroshot" in ours:
        uns = np.array([n not in seen for n in A["name"]])
        groups["seen names"], groups["unseen names"] = ~uns, uns
    print(f"\n== {ours} (xLSTM) vs {nomem} (no xLSTM)")
    for g, m in groups.items():
        a, n = pf1(A, ta, m); b, _ = pf1(B, tb, m)
        print(f"  {g:22s} n_gold={n:5d}  xLSTM {a:5.1f}  no-xLSTM {b:5.1f}  diff {a - b:+5.1f}")
