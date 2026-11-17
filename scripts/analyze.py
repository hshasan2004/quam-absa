"""Paper analysis for trained runs: confusion matrices, per-class / per-aspect / per-domain /
per-script tables, PR / ROC / threshold curves, error examples and efficiency numbers.

    python scripts/analyze.py eco_official_mm babsa_mm [...]

Re-evaluates runs/<run>/best.pt on the test split, using the detection threshold chosen on
validation (from metrics.json). Writes reports/analysis/<run>/: summary.md, summary.json,
per_aspect.csv, errors.md and figures as .png (preview) and .pdf (for LaTeX).
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from quam.data import OVERALL_QUERY, SCRIPTS, SENT_NAMES, ABSADataset, EncoderCache, load_records  # noqa: E402
from quam.metrics import babsa_metrics, eco_metrics, per_aspect_thresholds, prf  # noqa: E402
from quam.model import QuAM  # noqa: E402
from train import make_loader, to_dev  # noqa: E402

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- loading / inference
def load_run(name):
    run = ROOT / "runs" / name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return run, cfg, met


def build(cfg, cache, recs, eval_seed=None):
    """eval_seed fixes the sampled BABSA negatives (so several runs can be ensembled row by row)."""
    pool = sorted({a["name"] for r in recs if r["dataset"] == "babsa" and r["splits"].get("grouped") == "train"
                   for a in r["aspects"]})
    for r in recs:
        r["splits"]["_run"] = r["splits"][cfg["eco_split"]] if r["dataset"] == "eco" else r["splits"]["grouped"]
    kw = dict(babsa_pool=pool, babsa_neg=cfg["babsa_neg"], seed=cfg["seed"] if eval_seed is None else eval_seed)
    test_ds = ABSADataset(recs, cache, "_run", "test", **kw)
    val_ds = ABSADataset(recs, cache, "_run", "val", **kw)
    model = QuAM(cache.dim, cache.query(OVERALL_QUERY), cache.stats, d_model=cfg["d_model"], heads=cfg["heads"],
                 layout=cfg["layout"], dropout=cfg["dropout"], presence_mode=cfg["presence_mode"],
                 use_domain=not cfg["no_domain"], readout=cfg["readout"],
                 query_init=cfg.get("query_init", "free"))
    return test_ds, val_ds, model


@torch.no_grad()
def collect(model, loader, dev):
    """Per-aspect and per-review predictions plus timing of the forward pass alone."""
    model.eval()
    A, R = defaultdict(list), defaultdict(list)
    fwd_time, n_rev, n_asp = 0.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        b = to_dev(batch, dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        torch.cuda.synchronize(); fwd_time += time.perf_counter() - t0
        pres = out["presence"].float().sigmoid().cpu().numpy()
        sprob = out["sentiment"].float().softmax(-1).cpu().numpy()
        oprob = out["overall"].float().softmax(-1).cpu().numpy()
        ov = oprob.argmax(-1)
        am, scr = batch["a_mask"].numpy(), batch["script"].numpy()
        n_rev += len(batch["ids"]); n_asp += int(am.sum())
        for i in range(len(batch["ids"])):
            meta = dict(id=batch["ids"][i], dataset=batch["datasets"][i], domain=batch["domains"][i],
                        script=SCRIPTS[scr[i]])
            for j in np.nonzero(am[i])[0]:
                for k, v in meta.items():
                    A[k].append(v)
                A["name"].append(batch["aspect_names"][i][j])
                A["gold_pres"].append(batch["presence"][i, j].item())
                A["prob"].append(pres[i, j])
                A["gold_sent"].append(batch["sentiment"][i, j].item())
                A["pred_sent"].append(int(sprob[i, j].argmax()))
                A["sent_conf"].append(float(sprob[i, j].max()))
                A["sprob"].append(sprob[i, j])
                A["verbatim"].append(batch["verbatim"][i, j].item())
            for k, v in meta.items():
                R[k].append(v)
            R["gold"].append(batch["overall"][i].item())
            R["pred"].append(int(ov[i]))
            R["oprob"].append(oprob[i])
    eff = {"forward_seconds": fwd_time, "reviews": n_rev, "aspect_queries": n_asp,
           "reviews_per_s": n_rev / fwd_time, "ms_per_review": 1000 * fwd_time / n_rev,
           "peak_gpu_GB": torch.cuda.max_memory_allocated() / 2**30}
    return ({k: np.asarray(v) for k, v in A.items()}, {k: np.asarray(v) for k, v in R.items()}, eff)


# ---------------------------------------------------------------- plotting helpers
def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_cm(cm, labels, title, out, name):
    """Counts with row-normalised percentages (recall per gold class)."""
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 1.6, 1.15 * len(labels) + 1.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title)
    save(fig, out, name)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x):
    return "â€“" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def report_table(gold, pred, labels):
    rep = classification_report(gold, pred, labels=list(range(len(labels))), target_names=labels,
                                output_dict=True, zero_division=0)
    rows = [[n, pct(rep[n]["precision"]), pct(rep[n]["recall"]), pct(rep[n]["f1-score"]), int(rep[n]["support"])]
            for n in labels + ["macro avg", "weighted avg"]]
    return rep, md_table(["class", "P", "R", "F1", "support"], rows)


# ---------------------------------------------------------------- EComReviews
def eco_pair_stats(gold_pres, prob, gs, ps, t):
    gold, pred = gold_pres == 1, prob > t
    p, r, f = prf(int((pred & gold & (ps == gs)).sum()), int(pred.sum()), int(gold.sum()))
    dp, dr, df = prf(int((pred & gold).sum()), int(pred.sum()), int(gold.sum()))
    return dict(pair_P=p, pair_R=r, pair_F1=f, det_P=dp, det_R=dr, det_F1=df, n_gold=int(gold.sum()))


def analyze_eco(A, R, t, train_freq, texts, out, md, js):
    s = A["dataset"] == "eco"
    a = {k: v[s] for k, v in A.items()}
    rs = R["dataset"] == "eco"
    r = {k: v[rs] for k, v in R.items()}
    gold, prob = a["gold_pres"] == 1, a["prob"]
    pred = prob > t
    js["eco_metrics"] = eco_metrics(A, R, t)

    # 1) joint confusion: absent / positive / negative
    POS, NEG = 0, 2
    g3 = np.where(gold, np.where(a["gold_sent"] == NEG, 2, 1), 0)
    p3 = np.where(pred, np.where(a["pred_sent"] == NEG, 2, 1), 0)
    labels3 = ["absent", "positive", "negative"]
    cm = confusion_matrix(g3, p3, labels=[0, 1, 2])
    plot_cm(cm, labels3, f"Aspect-sentiment pairs (t={t:.2f})", out, "cm_eco_joint")
    js["cm_eco_joint"] = cm.tolist()

    # 2) detection
    cmd = confusion_matrix(gold, pred, labels=[False, True])
    plot_cm(cmd, ["absent", "present"], "Aspect detection", out, "cm_eco_detection")
    js["cm_eco_detection"] = cmd.tolist()

    # 3) polarity on gold-mentioned aspects (EComReviews has no neutral aspect label)
    gs, ps = a["gold_sent"][gold], a["pred_sent"][gold]
    gs2, ps2 = (gs == NEG).astype(int), (ps == NEG).astype(int)
    cms = confusion_matrix(gs2, ps2, labels=[0, 1])
    plot_cm(cms, ["positive", "negative"], "Aspect polarity (gold aspects)", out, "cm_eco_polarity")
    js["cm_eco_polarity"] = cms.tolist()
    rep_pol, tab_pol = report_table(gs2, ps2, ["positive", "negative"])
    js["report_eco_polarity"] = rep_pol

    # 4) overall review sentiment
    ok = r["gold"] >= 0
    cmo = confusion_matrix(r["gold"][ok], r["pred"][ok], labels=[0, 1, 2])
    plot_cm(cmo, SENT_NAMES, "Overall review sentiment", out, "cm_eco_overall")
    js["cm_eco_overall"] = cmo.tolist()
    rep_ov, tab_ov = report_table(r["gold"][ok], r["pred"][ok], SENT_NAMES)
    js["report_eco_overall"] = rep_ov

    # 5) per aspect
    rows = []
    for n in np.unique(a["name"]):
        m = a["name"] == n
        st = eco_pair_stats(a["gold_pres"][m], prob[m], a["gold_sent"][m], a["pred_sent"][m], t)
        g = gold & m
        st.update(aspect=n, train_freq=train_freq.get(n, 0), queried=int(m.sum()),
                  sent_acc=float((a["pred_sent"][g] == a["gold_sent"][g]).mean()) if g.any() else float("nan"))
        rows.append(st)
    rows.sort(key=lambda x: -x["n_gold"])
    with open(out / "per_aspect.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["aspect", "train_freq", "queried", "n_gold", "det_P", "det_R", "det_F1",
                                          "pair_P", "pair_R", "pair_F1", "sent_acc"])
        w.writeheader()
        w.writerows(rows)
    js["per_aspect"] = rows

    # frequency buckets (explains the low macro-F1) and seen vs unseen aspects (zero-shot)
    def bucket(fq):
        return "unseen (0)" if fq == 0 else "rare (1-49)" if fq < 50 else "mid (50-499)" if fq < 500 else "frequent (500+)"
    bucket_rows = []
    for bname in ["frequent (500+)", "mid (50-499)", "rare (1-49)", "unseen (0)"]:
        names = [x["aspect"] for x in rows if bucket(x["train_freq"]) == bname]
        if not names:
            continue
        m = np.isin(a["name"], names)
        st = eco_pair_stats(a["gold_pres"][m], prob[m], a["gold_sent"][m], a["pred_sent"][m], t)
        macro = [x["det_F1"] for x in rows if x["aspect"] in names and x["n_gold"] > 0]
        bucket_rows.append([bname, len(names), st["n_gold"], pct(st["det_F1"]), pct(st["pair_F1"]),
                            pct(np.mean(macro) if macro else float("nan"))])
    js["freq_buckets"] = bucket_rows

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = np.array([max(x["train_freq"], 0.7) for x in rows if x["n_gold"] > 0])
    ys = np.array([x["det_F1"] for x in rows if x["n_gold"] > 0])
    sz = np.array([x["n_gold"] for x in rows if x["n_gold"] > 0])
    ax.scatter(xs, ys, s=10 + 120 * sz / sz.max(), alpha=0.6, edgecolor="k", linewidth=0.4)
    ax.set_xscale("log"); ax.set_xlabel("training mentions of the aspect (log)"); ax.set_ylabel("detection F1")
    ax.set_title("Per-aspect detection F1 vs training frequency")
    save(fig, out, "aspect_f1_vs_freq")

    # 6) per domain and per script
    def group_rows(key):
        res = []
        for g in np.unique(a[key]):
            m = a[key] == g
            st = eco_pair_stats(a["gold_pres"][m], prob[m], a["gold_sent"][m], a["pred_sent"][m], t)
            rm = (r[key] == g) & (r["gold"] >= 0)
            ovf = f1_score(r["gold"][rm], r["pred"][rm], average="macro") if rm.any() else float("nan")
            res.append([g, int((r[key] == g).sum()), st["n_gold"], pct(st["det_F1"]), pct(st["pair_P"]),
                        pct(st["pair_R"]), pct(st["pair_F1"]), pct(ovf)])
        return sorted(res, key=lambda x: -x[1])
    dom_rows, scr_rows = group_rows("domain"), group_rows("script")
    js["per_domain"], js["per_script"] = dom_rows, scr_rows

    # 7) curves: detection PR, pair-F1 vs threshold
    prec, rec, _ = precision_recall_curve(gold, prob)
    ap = average_precision_score(gold, prob)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(rec, prec, lw=1.6)
    ax.scatter([js["eco_metrics"]["det_R"]], [js["eco_metrics"]["det_P"]], c="r", zorder=3, label=f"t={t:.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"Aspect detection PR (AP={ap:.3f})")
    ax.legend(); save(fig, out, "pr_eco_detection")
    grid = np.arange(0.05, 0.96, 0.05)
    sweep = [eco_pair_stats(a["gold_pres"], prob, a["gold_sent"], a["pred_sent"], g) for g in grid]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for k, lab in (("pair_P", "pair P"), ("pair_R", "pair R"), ("pair_F1", "pair F1")):
        ax.plot(grid, [x[k] for x in sweep], marker="o", ms=3, label=lab)
    ax.axvline(t, color="gray", ls="--", lw=1, label="val-chosen t")
    ax.set_xlabel("presence threshold"); ax.set_ylabel("score"); ax.set_title("Threshold sensitivity (test)")
    ax.legend(fontsize=8); save(fig, out, "threshold_sweep")
    js["detection_AP"] = ap
    js["threshold_sweep"] = [{"t": float(g), **x} for g, x in zip(grid, sweep)]

    # 8) error examples
    def ex(mask, order, k=12):
        idx = np.nonzero(mask)[0]
        idx = idx[np.argsort(order[idx])][:k]
        return [f"- **{a['name'][i]}** (p={prob[i]:.2f}, {a['domain'][i]}, {a['script'][i]}): {texts[a['id'][i]][:220]}"
                for i in idx]
    sent_err = gold & pred & (a["pred_sent"] != a["gold_sent"])
    errors = ["# Error examples (test)\n",
              "## Missed aspects (gold present, lowest probability)", *ex(gold & ~pred, prob),
              "\n## False alarms (gold absent, highest probability)", *ex(~gold & pred, -prob),
              "\n## Wrong polarity (detected, most confident mistakes)", *ex(sent_err, -a["sent_conf"])]
    (out / "errors.md").write_text("\n".join(errors) + "\n", encoding="utf-8")

    e = js["eco_metrics"]
    md += [f"## EComReviews-BD (threshold {t:.2f}, chosen on validation)\n",
           md_table(["pair P", "pair R", "pair F1", "det F1", "det macro-F1", "detection AP", "polarity acc",
                     "overall acc", "overall macro-F1"],
                    [[pct(e["pair_P"]), pct(e["pair_R"]), pct(e["pair_F1"]), pct(e["det_F1"]), pct(e["det_macroF1"]),
                      pct(ap), pct(e["sent_acc_on_gold"]), pct(e["overall_acc"]), pct(e["overall_macroF1"])]]),
           "\nFigures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, "
           "threshold_sweep, aspect_f1_vs_freq (.png / .pdf)\n",
           "### Joint confusion (rows gold, columns predicted)\n",
           md_table(["gold \\ pred", *labels3], [[labels3[i], *cm[i]] for i in range(3)]),
           "\n### Aspect polarity on gold-mentioned aspects\n", tab_pol,
           "\n### Overall review sentiment\n", tab_ov,
           "\n### By training frequency of the aspect\n",
           md_table(["bucket", "aspects", "test mentions", "det F1", "pair F1", "det macro-F1"], bucket_rows),
           "\n### By product category\n",
           md_table(["category", "reviews", "mentions", "det F1", "pair P", "pair R", "pair F1", "overall F1"], dom_rows),
           "\n### By script\n",
           md_table(["script", "reviews", "mentions", "det F1", "pair P", "pair R", "pair F1", "overall F1"], scr_rows),
           "\n### Per aspect: see per_aspect.csv. Top 15 by test mentions\n",
           md_table(["aspect", "train freq", "test mentions", "det F1", "pair F1", "polarity acc"],
                    [[x["aspect"], x["train_freq"], x["n_gold"], pct(x["det_F1"]), pct(x["pair_F1"]), pct(x["sent_acc"])]
                     for x in rows[:15]]), ""]


# ---------------------------------------------------------------- BABSA
def analyze_babsa(A, texts, out, md, js):
    s = A["dataset"] == "babsa"
    a = {k: v[s] for k, v in A.items()}
    js["babsa_metrics"] = babsa_metrics(A)
    g = a["gold_sent"] >= 0
    cm = confusion_matrix(a["gold_sent"][g], a["pred_sent"][g], labels=[0, 1, 2])
    plot_cm(cm, SENT_NAMES, "BABSA aspect sentiment", out, "cm_babsa_sentiment")
    js["cm_babsa_sentiment"] = cm.tolist()
    rep, tab = report_table(a["gold_sent"][g], a["pred_sent"][g], SENT_NAMES)
    js["report_babsa_sentiment"] = rep
    vt_rows = []
    for name, v in (("verbatim", 1), ("not verbatim (implicit)", 0)):
        m = g & (a["verbatim"] == v)
        c = confusion_matrix(a["gold_sent"][m], a["pred_sent"][m], labels=[0, 1, 2])
        tag = "verbatim" if v else "nonverbatim"
        plot_cm(c, SENT_NAMES, f"BABSA sentiment: {name}", out, f"cm_babsa_{tag}")
        js[f"cm_babsa_{tag}"] = c.tolist()
        vt_rows.append([name, int(m.sum()), pct((a["gold_sent"][m] == a["pred_sent"][m]).mean()),
                        pct(f1_score(a["gold_sent"][m], a["pred_sent"][m], average="macro"))])

    def group_rows(key):
        res = []
        for v in np.unique(a[key]):
            m = g & (a[key] == v)
            if m.sum() < 5:
                continue
            res.append([v, int(m.sum()), pct((a["gold_sent"][m] == a["pred_sent"][m]).mean()),
                        pct(f1_score(a["gold_sent"][m], a["pred_sent"][m], average="macro"))])
        return sorted(res, key=lambda x: -x[1])
    dom_rows, scr_rows = group_rows("domain"), group_rows("script")
    js["babsa_per_domain"], js["babsa_per_script"] = dom_rows, scr_rows

    pm = a["gold_pres"] >= 0
    fpr, tpr, _ = roc_curve(a["gold_pres"][pm], a["prob"][pm])
    auc = roc_auc_score(a["gold_pres"][pm], a["prob"][pm])
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot(fpr, tpr, lw=1.6); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"BABSA presence ROC (AUC={auc:.3f})"); save(fig, out, "roc_babsa_presence")

    idx = np.nonzero(g & (a["pred_sent"] != a["gold_sent"]))[0]
    idx = idx[np.argsort(-a["sent_conf"][idx])][:20]
    errors = ["# BABSA error examples (most confident wrong sentiment)\n"]
    errors += [f"- **{a['name'][i]}** gold={SENT_NAMES[a['gold_sent'][i]]} pred={SENT_NAMES[a['pred_sent'][i]]} "
               f"(conf {a['sent_conf'][i]:.2f}, {a['domain'][i]}, verbatim={a['verbatim'][i]}): {texts[a['id'][i]][:220]}"
               for i in idx]
    (out / "errors.md").write_text("\n".join(errors) + "\n", encoding="utf-8")

    b = js["babsa_metrics"]
    md += ["## BABSA\n",
           md_table(["sent acc", "sent macro-F1", "verbatim F1", "implicit F1", "presence AUROC"],
                    [[pct(b["sent_acc"]), pct(b["sent_macroF1"]), pct(b.get("sent_macroF1_verbatim")),
                      pct(b.get("sent_macroF1_nonverbatim")), pct(auc)]]),
           "\nFigures: cm_babsa_sentiment, cm_babsa_verbatim, cm_babsa_nonverbatim, roc_babsa_presence\n",
           "### Confusion (rows gold, columns predicted)\n",
           md_table(["gold \\ pred", *SENT_NAMES], [[SENT_NAMES[i], *cm[i]] for i in range(3)]),
           "\n### Per class\n", tab,
           "\n### Verbatim vs implicit aspect terms\n", md_table(["aspects", "n", "acc", "macro-F1"], vt_rows),
           "\n### By domain\n", md_table(["domain", "n", "acc", "macro-F1"], dom_rows),
           "\n### By script\n", md_table(["script", "n", "acc", "macro-F1"], scr_rows), ""]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    dev = "cuda"
    cache = None
    for name in args.runs:
        run, cfg, met = load_run(name)
        if cache is None:
            from quam.data import default_cache_dir
            cache = EncoderCache(cfg.get("cache") or default_cache_dir())
        which = {"eco": ("eco",), "babsa": ("babsa",), "both": ("eco", "babsa")}[cfg["data"]]
        recs = load_records(which)
        texts = {r["id"]: r["text"] for r in recs}
        test_ds, val_ds, model = build(cfg, cache, recs)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        model.to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        A, R, eff = collect(model, make_loader(test_ds, 64, False, 0), dev)

        out = ROOT / "reports" / "analysis" / name
        out.mkdir(parents=True, exist_ok=True)
        heads, inner = cfg["heads"], 2 * cfg["d_model"] // cfg["heads"]
        n_blocks = cfg["layout"].count("m")
        eff.update(params_M=n_params / 1e6,
                   memory_floats_per_review=2 * heads * inner * (inner + 1),
                   note="memory = final mLSTM state (C, n) of both directions; fixed size, independent of review "
                        "length and of the number of aspects queried")
        js = {"run": name, "config": cfg, "efficiency": eff, "best_epoch": met["best_epoch"]}
        md = [f"# Analysis: {name}\n",
              f"split `{cfg['eco_split'] if cfg['data'] != 'babsa' else 'grouped'}`, layout `{cfg['layout']}`, "
              f"readout `{cfg['readout']}`, presence `{cfg['presence_mode']}`, seed {cfg['seed']}, "
              f"best epoch {met['best_epoch']}\n",
              "## Efficiency (test set, forward pass only, bf16, RTX 2050)\n",
              md_table(["params", "reviews/s", "ms/review", "aspect queries", "peak GPU",
                        "memory per review"],
                       [[f"{n_params / 1e6:.2f}M", f"{eff['reviews_per_s']:.0f}", f"{eff['ms_per_review']:.2f}",
                         f"{eff['aspect_queries']:,}", f"{eff['peak_gpu_GB']:.2f} GB",
                         f"{eff['memory_floats_per_review']:,} floats ({n_blocks} mLSTM blocks, last one queried)"]]),
              ""]
        if "eco" in met["test"]:
            freq = Counter(a["query"] for r in recs if r["dataset"] == "eco" and r["splits"]["_run"] == "train"
                           for a in r["aspects"] if a["label"] != "none")
            t = met["test"]["eco"]["threshold"]
            analyze_eco(A, R, t, freq, texts, out, md, js)
            # per-aspect thresholds, tuned on validation only
            Av, _, _ = collect(model, make_loader(val_ds, 64, False, 0), dev)
            thr = per_aspect_thresholds(Av, t)
            e1, e2 = js["eco_metrics"], eco_metrics(A, R, thr)
            js["eco_metrics_per_aspect_threshold"], js["per_aspect_thresholds"] = e2, thr
            md += ["## Per-aspect thresholds (tuned on validation, applied to test)\n",
                   f"{len(thr) - 1} aspects with at least 50 validation mentions get their own threshold; "
                   f"the rest keep the global one ({t:.2f}).\n",
                   md_table(["threshold", "pair P", "pair R", "pair F1", "det F1", "det macro-F1"],
                            [[name, pct(e["pair_P"]), pct(e["pair_R"]), pct(e["pair_F1"]), pct(e["det_F1"]),
                              pct(e["det_macroF1"])] for name, e in (("global", e1), ("per-aspect", e2))]), ""]
        if "babsa" in met["test"]:
            analyze_babsa(A, texts, out, md, js)
        (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(js, indent=1, default=lambda x: x.item() if hasattr(x, "item")
                                                     else str(x)), encoding="utf-8")
        print(f"{name}: wrote {out.relative_to(ROOT)}  ({eff['reviews_per_s']:.0f} reviews/s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
