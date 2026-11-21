"""Ablation table for the final model (aspect-prompted QuAM, fine-tuned XLM-R, Gated DeltaNet memory).

    python scripts/ablation_table.py      -> prints a table, writes reports/ablation.json
Mean over the seeds that exist (n). Zero-shot also reports pair F1 on unseen aspect names.
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

ROWS = [  # (label, official run, zeroshot run)
    ("Full model (prompt + fine-tuned XLM-R + Gated DeltaNet)", "ftp_gdn_official", "ftp_gdn_zeroshot"),
    ("  aspect read: marker token (TMM-style) instead of span mean", "ftp_gdn_marker_official", "ftp_gdn_marker_zeroshot"),
    ("  memory: xLSTM (mLSTM) instead of Gated DeltaNet", "ftp_official", "ftp_zeroshot"),
    ("  memory: none (direct scorer only)", "ftp_nomem_official", "ftp_nomem_zeroshot"),
    ("  memory only (no direct scorer), xLSTM", None, "ftp_memonly_zeroshot"),
    ("  no aspect prompt (xLSTM)", "ft_official", "ft_zeroshot"),
    ("  no aspect prompt, no memory", "ft_nomem_official", "ft_nomem_zeroshot"),
    ("  encoder 9/12 layers frozen (xLSTM)", "ftp_f9_official", None),
    ("  encoder 9/12 layers frozen, no memory", "ftp_nomem_f9_official", None),
    ("  encoder fully frozen (xLSTM)", "ftp_f12_official", None),
    ("  encoder fully frozen, no memory", "ftp_nomem_f12_official", None),
    ("  old QuAM: frozen cached XLM-R, no prompt (xLSTM)", "q2_official", "q2_zeroshot"),
]
recs = load_records(("eco",))
seen = {a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}


def seeds(b):
    return [n for n in (b, b + "_s1", b + "_s2") if (ROOT / "runs" / n / "metrics.json").exists()]


def unseen_pair(n):
    p = ROOT / "runs" / n / "test_preds.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    A, t = {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])
    m = np.array([a not in seen for a in A["name"]])
    pred, gold = (A["prob"] > t) & m, (A["gold_pres"] == 1) & m
    return 200 * (pred & gold & (A["pred_sent"] == A["gold_sent"])).sum() / max(pred.sum() + gold.sum(), 1)


def agg(base, keys, unseen=False):
    if not base or not seeds(base):
        return None
    out = {"n": len(seeds(base))}
    for k in keys:
        v = [100 * json.load(open(ROOT / "runs" / n / "metrics.json"))["test"]["eco"][k] for n in seeds(base)]
        out[k] = (st.mean(v), st.stdev(v) if len(v) > 1 else None)
    if unseen:
        v = [u for u in (unseen_pair(n) for n in seeds(base)) if u is not None]
        if not v and (ROOT / "reports" / "unseen_aspects.json").exists():   # frozen QuAM: from the analysis report
            r = json.load(open(ROOT / "reports" / "unseen_aspects.json")).get(base)
            v = [r["unseen_pair_F1"]["mean"]] if r else []
        out["unseen"] = (st.mean(v), None) if v else None
    return out


K = ("pair_F1", "det_macroF1", "overall_macroF1")
res = []
fmt = lambda x: "  -  " if x is None else f"{x[0]:5.1f}"
print(f"{'':58s} | {'Official':^26s} | {'New categories (zero-shot)':^34s}")
print(f"{'Variant':58s} | {'n':>2s} {'pair':>5s} {'macro':>5s} {'overall':>7s} | {'n':>2s} {'pair':>5s} {'macro':>5s} {'unseen':>6s} {'overall':>7s}")
for lab, o, z in ROWS:
    a, b = agg(o, K), agg(z, K, unseen=True)
    res.append({"variant": lab.strip(), "official": a, "zeroshot": b})
    so = f"{a['n']:2d} {fmt(a['pair_F1'])} {fmt(a['det_macroF1'])} {fmt(a['overall_macroF1']):>7s}" if a else " " * 26
    sz = (f"{b['n']:2d} {fmt(b['pair_F1'])} {fmt(b['det_macroF1'])} {fmt(b['unseen']):>6s} {fmt(b['overall_macroF1']):>7s}"
          if b else " " * 34)
    print(f"{lab:58s} | {so} | {sz}")
(ROOT / "reports" / "ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
