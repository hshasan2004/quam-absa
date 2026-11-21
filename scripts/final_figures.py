"""Figures and analysis numbers of the final model from its test predictions -> paper_journal/figures/, reports/final_analysis.json.

    python scripts/final_figures.py
threshold sweep (official, seed 42), joint confusion matrix (official, seed 42), BABSA confusion
matrix (seed 42), per-aspect detection F1 vs training frequency (grouped, 3 seeds averaged).
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

OUT = ROOT / "paper_journal" / "figures"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
POS, NEG = 0, 2


def load(n):
    z = np.load(ROOT / "runs" / n / "test_preds.npz", allow_pickle=True)
    return {k[2:]: z[k] for k in z.files if k.startswith("A_")}, float(z["threshold"])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)


def cm_plot(cm, labels, title, name):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 1.6, 1.15 * len(labels) + 1.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title)
    save(fig, name)


res = {}
# threshold sweep
A, t = load("ftp_gdn_official")
gold = A["gold_pres"] == 1; ok = A["pred_sent"] == A["gold_sent"]
ts = np.round(np.arange(0.05, 0.96, 0.05), 2); rows = []
for th in ts:
    pred = A["prob"] > th
    tp, dtp = (pred & gold & ok).sum(), (pred & gold).sum()
    P, R = tp / max(pred.sum(), 1), tp / gold.sum(); dP, dR = dtp / max(pred.sum(), 1), dtp / gold.sum()
    rows.append((th, P, R, 2 * P * R / max(P + R, 1e-9), 2 * dP * dR / max(dP + dR, 1e-9)))
rows = np.array(rows); res["sweep"] = {f"{r[0]:.2f}": [100 * x for x in r[1:]] for r in rows}; res["sweep_val_threshold"] = t
fig, ax = plt.subplots(figsize=(5, 3.2))
for j, lab in ((1, "pair P"), (2, "pair R"), (3, "pair F1"), (4, "detection F1")):
    ax.plot(rows[:, 0], 100 * rows[:, j], marker="o", ms=3, label=lab, ls="--" if j == 4 else "-")
ax.axvline(t, color="gray", lw=0.8, ls=":"); ax.text(t, 5, " val. threshold", fontsize=8, color="gray")
ax.set_xlabel("presence threshold"); ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend(fontsize=8, ncol=2)
ax.set_title("Threshold sweep (official split, test, seed 42)"); save(fig, "final_threshold_sweep")

# joint confusion (absent / pos / neg)
pred = A["prob"] > t
g = np.where(~gold, 0, np.where(A["gold_sent"] == POS, 1, 2)); p = np.where(~pred, 0, np.where(A["pred_sent"] == POS, 1, 2))
cm = np.zeros((3, 3)); np.add.at(cm, (g, p), 1)
cm_plot(cm, ["absent", "positive", "negative"], "EComReviews-BD (official, seed 42)", "final_cm_eco")
res["cm_eco"] = cm.tolist()
res["neg_missed"] = cm[2, 0] / cm[2].sum(); res["pos_missed"] = cm[1, 0] / cm[1].sum()
res["neg_pol_correct"] = [int(cm[2, 2]), int(cm[2, 1] + cm[2, 2])]; res["pos_pol_correct"] = [int(cm[1, 1]), int(cm[1, 1] + cm[1, 2])]

# BABSA confusion
B, _ = load("ftp_gdn_babsa_banglabert"); m = B["gold_pres"] == 1
cmb = np.zeros((3, 3)); np.add.at(cmb, (B["gold_sent"][m], B["pred_sent"][m]), 1)
cm_plot(cmb, ["positive", "neutral", "negative"], "BABSA (seed 42)", "final_cm_babsa"); res["cm_babsa"] = cmb.tolist()

# per-aspect detection F1 vs training frequency (grouped split)
recs = load_records(("eco",))
freq = Counter(a["query"] for r in recs if r["splits"]["grouped"] == "train" for a in r["aspects"] if a["label"] != "none")
F = {}
for n in ("ftp_gdn_grouped", "ftp_gdn_grouped_s1", "ftp_gdn_grouped_s2", "bl_banglabert_grouped", "bl_banglabert_grouped_s1", "bl_banglabert_grouped_s2"):
    A, t = load(n); pred, gold = A["prob"] > t, A["gold_pres"] == 1
    if "name" not in A:
        continue
    for a in set(A["name"]):
        s = A["name"] == a
        if gold[s].sum():
            F.setdefault(n.split("_grouped")[0], {}).setdefault(a, []).append(200 * (pred & gold & s).sum() / max((pred & s).sum() + (gold & s).sum(), 1))
ours = {a: np.mean(v) for a, v in F["ftp_gdn"].items()}
ngold = Counter(A["name"][A["gold_pres"] == 1])
buckets = {"frequent (500+)": lambda f: f >= 500, "mid (50-499)": lambda f: 50 <= f < 500, "rare (<50)": lambda f: f < 50}
res["freq_buckets"] = {b: {"n_aspects": sum(fn(freq[a]) for a in ours), "mean_det_F1": float(np.mean([v for a, v in ours.items() if fn(freq[a])]))} for b, fn in buckets.items()}
fig, ax = plt.subplots(figsize=(5, 3.2))
xs = [max(freq[a], 0.7) for a in ours]; ax.scatter(xs, list(ours.values()), s=[8 + ngold[a] / 4 for a in ours], alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("training mentions"); ax.set_ylabel("detection F1 (%)"); ax.set_ylim(-3, 100)
ax.set_title("Per-aspect detection F1 (grouped split, mean of 3 seeds)"); save(fig, "final_aspect_f1_vs_freq")
(ROOT / "reports" / "final_analysis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1, default=float)); print({k: [round(x, 1) for x in v] for k, v in res["sweep"].items() if k in ("0.05", "0.10", "0.50")})
