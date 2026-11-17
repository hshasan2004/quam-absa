"""Evaluation metrics.

EComReviews (all category aspects are queried for every review):
  detection   - is the aspect mentioned? micro P/R/F1 and macro-F1 over aspect names
  pair strict - (aspect, polarity) must both be right; micro P/R/F1  (the paper's main metric)
  sentiment   - polarity accuracy on gold-mentioned aspects
  overall     - review-level sentiment accuracy / macro-F1
BABSA (gold aspects plus sampled absent aspects):
  sentiment   - 3-class accuracy / macro-F1 on gold aspects, also split verbatim vs not
  presence    - AUROC of gold aspects vs sampled absent ones
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


class Collector:
    """Accumulates per-aspect and per-review predictions over an evaluation pass."""

    def __init__(self):
        self.asp = defaultdict(list)
        self.rev = defaultdict(list)

    def add(self, batch, out):
        pres = out["presence"].float().sigmoid().cpu().numpy()
        sent = out["sentiment"].float().argmax(-1).cpu().numpy()
        overall = out["overall"].float().argmax(-1).cpu().numpy()
        a_mask = batch["a_mask"].numpy()
        for i in range(len(batch["ids"])):
            ds = batch["datasets"][i]
            for j in np.nonzero(a_mask[i])[0]:
                self.asp["dataset"].append(ds)
                self.asp["domain"].append(batch["domains"][i])
                self.asp["name"].append(batch["aspect_names"][i][j])
                self.asp["gold_pres"].append(batch["presence"][i, j].item())
                self.asp["prob"].append(pres[i, j])
                self.asp["gold_sent"].append(batch["sentiment"][i, j].item())
                self.asp["pred_sent"].append(sent[i, j])
                self.asp["verbatim"].append(batch["verbatim"][i, j].item())
            self.rev["dataset"].append(ds)
            self.rev["gold"].append(batch["overall"][i].item())
            self.rev["pred"].append(overall[i])

    def arrays(self):
        return ({k: np.asarray(v) for k, v in self.asp.items()}, {k: np.asarray(v) for k, v in self.rev.items()})


def eco_metrics(a, rv, threshold):
    sel = a["dataset"] == "eco"
    if not sel.any():
        return {}
    gold, prob, gs, ps, names = a["gold_pres"][sel] == 1, a["prob"][sel], a["gold_sent"][sel], a["pred_sent"][sel], a["name"][sel]
    if isinstance(threshold, dict):            # per-aspect thresholds {name: t, "_default": t}
        t_rows = np.array([threshold.get(n, threshold["_default"]) for n in names])
        pred = prob > t_rows
        threshold = "per-aspect"
    else:
        pred = prob > threshold
    d_p, d_r, d_f = prf(int((pred & gold).sum()), int(pred.sum()), int(gold.sum()))
    macro = [f1_score(gold[names == n], pred[names == n], zero_division=0) for n in np.unique(names) if gold[names == n].any()]
    tp_pair = int((pred & gold & (ps == gs)).sum())
    p_p, p_r, p_f = prf(tp_pair, int(pred.sum()), int(gold.sum()))
    rsel = rv["dataset"] == "eco"
    return {
        "threshold": threshold,
        "det_P": d_p, "det_R": d_r, "det_F1": d_f, "det_macroF1": float(np.mean(macro)),
        "pair_P": p_p, "pair_R": p_r, "pair_F1": p_f,
        "sent_acc_on_gold": float((ps[gold] == gs[gold]).mean()),
        "overall_acc": accuracy_score(rv["gold"][rsel], rv["pred"][rsel]),
        "overall_macroF1": f1_score(rv["gold"][rsel], rv["pred"][rsel], average="macro"),
    }


def best_eco_threshold(a, rv, grid=np.arange(0.05, 0.96, 0.05)):
    scores = [(eco_metrics(a, rv, float(t)).get("pair_F1", 0.0), float(t)) for t in grid]
    return max(scores)[1]


def per_aspect_thresholds(a, global_t, min_gold=50, grid=np.arange(0.05, 0.96, 0.05)):
    """Tune one detection threshold per aspect on validation predictions.

    Aspects with fewer than `min_gold` validation mentions keep the global threshold.
    Each threshold maximises that aspect's strict pair F1 (aspect + polarity).
    """
    sel = a["dataset"] == "eco"
    names, gold, prob = a["name"][sel], a["gold_pres"][sel] == 1, a["prob"][sel]
    ok = a["pred_sent"][sel] == a["gold_sent"][sel]
    thr = {"_default": float(global_t)}
    for n in np.unique(names):
        m = names == n
        if gold[m].sum() < min_gold:
            continue
        def f1_at(t):
            pred = prob[m] > t
            return prf(int((pred & gold[m] & ok[m]).sum()), int(pred.sum()), int(gold[m].sum()))[2]
        # ties are broken towards the global threshold (less overfitting to validation)
        thr[str(n)] = max((f1_at(t), -abs(t - global_t), float(t)) for t in grid)[2]
    return thr


def babsa_metrics(a):
    sel = a["dataset"] == "babsa"
    if not sel.any():
        return {}
    gold = sel & (a["gold_sent"] >= 0)
    out = {
        "sent_acc": accuracy_score(a["gold_sent"][gold], a["pred_sent"][gold]),
        "sent_macroF1": f1_score(a["gold_sent"][gold], a["pred_sent"][gold], average="macro"),
    }
    for name, v in (("verbatim", 1), ("nonverbatim", 0)):
        g = gold & (a["verbatim"] == v)
        if g.any():
            out[f"sent_macroF1_{name}"] = f1_score(a["gold_sent"][g], a["pred_sent"][g], average="macro")
            out[f"n_{name}"] = int(g.sum())
    pres = sel & (a["gold_pres"] >= 0)
    if len(np.unique(a["gold_pres"][pres])) == 2:
        out["presence_AUROC"] = roc_auc_score(a["gold_pres"][pres], a["prob"][pres])
    return out


def per_group(a, key, dataset="eco", threshold=0.5):
    """Pair-F1 (eco) or sentiment macro-F1 (babsa) per value of `key` (e.g. domain)."""
    res = {}
    for g in np.unique(a[key][a["dataset"] == dataset]):
        s = (a["dataset"] == dataset) & (a[key] == g)
        if dataset == "eco":
            gold, pred = a["gold_pres"][s] == 1, a["prob"][s] > threshold
            res[str(g)] = prf(int((pred & gold & (a["pred_sent"][s] == a["gold_sent"][s])).sum()), int(pred.sum()), int(gold.sum()))[2]
        else:
            gg = s & (a["gold_sent"] >= 0)
            res[str(g)] = f1_score(a["gold_sent"][gg], a["pred_sent"][gg], average="macro") if gg.any() else None
    return res
