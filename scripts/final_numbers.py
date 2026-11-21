"""All numbers of the final paper (aspect-prompted QuAM with Gated DeltaNet memory) -> reports/final_numbers.json.

    python scripts/final_numbers.py
Per system: mean/std over the seeds that exist of every test metric; on the zero-shot split also
seen/unseen-name detection and pair F1; on the official split pair F1 by script; on BABSA accuracy
and macro-F1 (all, verbatim, implicit).
"""
import json, statistics as st, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

ECO = {"ours": "ftp_gdn", "ours_xlstm": "ftp", "ours_nomem": "ftp_nomem", "banglabert": "bl_banglabert",
       "xlmr": "bl_xlmr", "xlmr_frozen": "bl_xlmr_frozen", "quam_frozen": "q2"}
BABSA = {"ours": "ftp_gdn_babsa_banglabert", "ours_xlmr": "ftp_gdn_babsa_xlmr",
         "ours_xlstm": "ftp_babsa_banglabert", "banglabert": "bl_babsa_banglabert", "xlmr": "bl_babsa_xlmr",
         "xlmr_frozen": "bl_babsa_xlmr_frozen", "quam_frozen": "q2_babsa"}
recs = {r["id"]: r for r in load_records(("eco",))}
seen = {a["query"] for r in recs.values() if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}


def seeds(b):
    return [n for n in (b, b + "_s1", b + "_s2") if (ROOT / "runs" / n / "metrics.json").exists()]


def ms(v):
    v = [x for x in v if x is not None]
    return None if not v else {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else None, "n": len(v)}


def preds(n):
    p = ROOT / "runs" / n / "test_preds.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"]) if "threshold" in z.files else None


def sub(A, t, m):
    pred, gold = (A["prob"] > t) & m, (A["gold_pres"] == 1) & m
    tp_d = (pred & gold).sum(); tp = (tp_d if False else (pred & gold & (A["pred_sent"] == A["gold_sent"])).sum())
    d = max(pred.sum() + gold.sum(), 1)
    return 200 * tp_d / d, 200 * tp / d


out = {"eco": {}, "babsa": {}}
for split in ("official", "grouped", "zeroshot"):
    for k, b in ECO.items():
        ns = seeds(f"{b}_{split}")
        if not ns:
            continue
        M = [json.load(open(ROOT / "runs" / n / "metrics.json"))["test"]["eco"] for n in ns]
        res = {m: ms([100 * x[m] for x in M]) for m in M[0] if m != "threshold"}
        res["runs"] = ns
        P = [preds(n) for n in ns]
        if all(p and "name" in p[0] for p in P):
            if split == "zeroshot":
                for lab, f in (("seen", True), ("unseen", False)):
                    v = [sub(A, t, np.array([(a in seen) == f for a in A["name"]])) for A, t in P]
                    res[f"{lab}_det_F1"], res[f"{lab}_pair_F1"] = ms([x[0] for x in v]), ms([x[1] for x in v])
            if split == "official":
                for s in ("english", "bangla", "mixed", "banglish"):
                    res[f"script_{s}"] = ms([sub(A, t, np.array([recs[i]["script"] == s for i in A["id"]]))[1] for A, t in P])
        out["eco"].setdefault(split, {})[k] = res
for k, b in BABSA.items():
    ns = seeds(b)
    if ns:
        M = [json.load(open(ROOT / "runs" / n / "metrics.json"))["test"]["babsa"] for n in ns]
        out["babsa"][k] = {m: ms([100 * x[m] for x in M]) for m in ("sent_acc", "sent_macroF1", "sent_macroF1_verbatim", "sent_macroF1_nonverbatim")}
        out["babsa"][k]["runs"] = ns
(ROOT / "reports" / "final_numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
f = lambda d: "-" if not d else (f"{d['mean']:.1f}" + (f"±{d['std']:.1f}" if d["std"] is not None else "") + f"(n{d['n']})")
for split, D in out["eco"].items():
    print("==", split)
    for k, r in D.items():
        print(f"{k:12s}", " ".join(f"{m}={f(r.get(m))}" for m in ("pair_P", "pair_R", "pair_F1", "det_F1", "det_macroF1", "sent_acc_on_gold", "overall_macroF1")),
              " ".join(f"{m}={f(v)}" for m, v in r.items() if m.startswith(("seen", "unseen", "script"))))
print("== babsa")
for k, r in out["babsa"].items():
    print(f"{k:12s}", " ".join(f"{m}={f(v)}" for m, v in r.items() if m != "runs"))
