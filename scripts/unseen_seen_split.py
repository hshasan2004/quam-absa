import json, sys, numpy as np
from pathlib import Path
ROOT = Path(r"C:/Users/USER/OneDrive/Desktop/new/quam-absa"); sys.path.insert(0, str(ROOT))
from quam.data import load_records
recs = load_records(("eco",))
seen = {a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"}
def f1(tp, p, g): return 2 * tp / max(p + g, 1)
for n in sys.argv[1:]:
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    A = {k[2:]: z[k] for k in z.files if k.startswith("A_")}; t = float(z["threshold"])
    pred, gold = A["prob"] > t, A["gold_pres"] == 1
    ok = pred & gold & (A["pred_sent"] == A["gold_sent"])
    out = []
    for grp in (True, False):
        m = np.array([(x not in seen) == grp for x in A["name"]])
        out += [100 * f1((ok & m).sum(), (pred & m).sum(), (gold & m).sum()), 100 * f1((pred & gold & m).sum(), (pred & m).sum(), (gold & m).sum())]
    print(f"{n:22s} unseen pair {out[0]:.1f} det {out[1]:.1f} | seen pair {out[2]:.1f} det {out[3]:.1f}")
