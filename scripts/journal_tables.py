"""Extra numbers for the journal version: dataset statistics and per-script / per-category pair F1
of QuAM v2 and the baselines (official split, mean over the available seeds).

    python scripts/journal_tables.py --device cpu       -> reports/journal_tables.json
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze import build, cache_for, collect, load_run  # noqa: E402
from quam.data import load_records  # noqa: E402
from train import make_loader  # noqa: E402

SYSTEMS = {"quam": "q2_official", "banglabert_ft": "bl_banglabert_official", "xlmr_frozen": "bl_xlmr_frozen_official",
           "xlmr_ft": "bl_xlmr_official"}


def pair_f1(A, t, mask):
    pred, gold = (A["prob"] > t) & mask, (A["gold_pres"] == 1) & mask
    tp = (pred & gold & (A["pred_sent"] == A["gold_sent"])).sum()
    return 200 * tp / max(pred.sum() + gold.sum(), 1)


def preds(name, recs, dev):
    if (ROOT / "runs" / name / "best.pt").exists():
        run, cfg, met = load_run(name)
        test_ds, _, model = build(cfg, cache_for(cfg), recs)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        A, _, _ = collect(model.to(dev), make_loader(test_ds, 64, False, 0), dev)
        return A, met["test"]["eco"]["threshold"]
    z = np.load(ROOT / "runs" / name / "test_preds.npz", allow_pickle=True)
    return {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    dev = ap.parse_args().device
    eco, bab = load_records(("eco",)), load_records(("babsa",))
    out = {"stats": {
        "eco_script": Counter(r["script"] for r in eco), "babsa_script": Counter(r["script"] for r in bab),
        "eco_category": Counter(r["domain"] for r in eco), "babsa_domain": Counter(r["domain"] for r in bab),
        "eco_overall": Counter(r["overall"] for r in eco),
        "eco_aspect_labels": Counter(a["label"] for r in eco for a in r["aspects"]),
        "babsa_aspect_labels": Counter(a["label"] for r in bab for a in r["aspects"]),
        "eco_words_median": statistics.median(len(r["text"].split()) for r in eco if r["text"]),
        "babsa_words_median": statistics.median(len(r["text"].split()) for r in bab if r["text"]),
    }}
    script = {r["id"]: r["script"] for r in eco}
    res = {}
    for key, base in SYSTEMS.items():
        seeds = [n for n in (base, f"{base}_s1", f"{base}_s2") if (ROOT / "runs" / n / "metrics.json").exists()]
        per = {"script": {}, "category": {}}
        for n in seeds:
            A, t = preds(n, eco, dev)
            sc = np.array([script[i] for i in A["id"]])
            for s in ("english", "bangla", "banglish", "mixed"):
                per["script"].setdefault(s, []).append(pair_f1(A, t, sc == s))
            for c in np.unique(A["domain"]):
                per["category"].setdefault(str(c), []).append(pair_f1(A, t, A["domain"] == c))
        res[key] = {"n_seeds": len(seeds), **{g: {k: statistics.mean(v) for k, v in d.items()} for g, d in per.items()}}
        print(key, len(seeds), {k: round(v, 1) for k, v in res[key]["script"].items()})
    out["official_by_group"] = res
    (ROOT / "reports" / "journal_tables.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out["stats"], indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
