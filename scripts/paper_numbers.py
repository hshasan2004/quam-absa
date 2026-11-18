"""Collect every number used in the paper into paper/numbers.json (mean/std over seeds, ensembles)."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
ECO = ["pair_P", "pair_R", "pair_F1", "det_F1", "det_macroF1", "sent_acc_on_gold", "overall_acc", "overall_macroF1"]
BAB = ["sent_acc", "sent_macroF1", "sent_macroF1_verbatim", "sent_macroF1_nonverbatim", "presence_AUROC"]


def load(name):
    f = ROOT / "runs" / name / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def group(base, keys, part):
    runs = [load(n) for n in (base, f"{base}_s1", f"{base}_s2")]
    runs = [r["test"][part] for r in runs if r]
    keys = [k for k in keys if runs and all(k in r for r in runs)]
    out = {"n": len(runs)}
    for k in keys:
        v = [100 * r[k] for r in runs]
        out[k] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0, "runs": v}
    return out


def single(name, keys, part):
    r = load(name)
    return {k: 100 * r["test"][part][k] for k in keys if k in r["test"][part]} if r else None


def main():
    N = {"seeded": {}, "ensemble": {}, "single": {}, "analysis": {}}
    for b in ["hyb_official", "fin_hybrid", "hyb_zeroshot", "fin_meanpool", "fin_memory", "fin_tied", "fin_pres_val"]:
        N["seeded"][b] = group(b, ECO, "eco")
    N["seeded"]["hyb_babsa"] = group("hyb_babsa", BAB, "babsa")
    # QuAM v2, its no-memory ablation, and our own baselines (3 seeds where available)
    for b in ["q2_official", "q2_grouped", "q2_zeroshot", "q2only_grouped", "q2only_zeroshot",
              "bl_banglabert_official", "bl_banglabert_grouped", "bl_banglabert_zeroshot",
              "bl_xlmr_frozen_official", "bl_xlmr_frozen_grouped", "bl_xlmr_frozen_zeroshot",
              "bl_xlmr_official", "bl_xlmr_grouped", "bl_xlmr_zeroshot",
              "bl_banglabert_frozen_official", "bl_banglabert_frozen_grouped"]:
        N["seeded"][b] = group(b, ECO, "eco")
    for b in ["q2_babsa", "bl_babsa_banglabert", "bl_babsa_xlmr_frozen", "bl_babsa_xlmr"]:
        N["seeded"][b] = group(b, [k for k in BAB if k != "presence_AUROC" or b == "q2_babsa"], "babsa")
    N["significance"] = {f.stem.replace("significance_", ""): json.loads(f.read_text(encoding="utf-8"))
                         for f in sorted((ROOT / "reports").glob("significance_*.json"))}
    for e in ["ens_hyb_official", "ens_hybrid", "ens_hyb_zeroshot", "ens_meanpool", "ens_memory",
              "ens_q2_official", "ens_q2_grouped", "ens_q2_zeroshot"]:
        N["ensemble"][e] = single(e, ECO, "eco")
    N["ensemble"]["ens_hyb_babsa"] = single("ens_hyb_babsa", BAB, "babsa")
    N["ensemble"]["ens_q2_babsa"] = single("ens_q2_babsa", BAB, "babsa")
    for s in ["eco_official_mm", "eco_official_full", "eco_grouped_mm", "eco_grouped_full", "abl_nodomain",
              "abl_depth3", "eco_grouped_cons", "abl_pres_mag", "abl_pres_val", "abl_meanpool", "eco_zeroshot_mm"]:
        N["single"][s] = single(s, ECO, "eco")
    N["single"]["babsa_mm"] = single("babsa_mm", BAB, "babsa")
    for a in ["hyb_official", "fin_hybrid", "hyb_zeroshot", "hyb_babsa", "eco_zeroshot_mm", "fin_memory",
              "q2_official", "q2_grouped", "q2_zeroshot", "q2_babsa"]:
        f = ROOT / "reports" / "analysis" / a / "summary.json"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            N["analysis"][a] = {k: j.get(k) for k in ["efficiency", "cm_eco_joint", "cm_eco_overall", "cm_eco_polarity",
                                                     "freq_buckets", "per_script", "per_domain", "detection_AP",
                                                     "cm_babsa_sentiment", "babsa_per_script", "babsa_per_domain",
                                                     "eco_metrics_per_aspect_threshold", "eco_metrics"]}
    out = ROOT / "paper" / "numbers.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    # compact print
    for b, g in N["seeded"].items():
        print(b, {k: f"{v['mean']:.1f}±{v['std']:.1f}" for k, v in g.items() if k != "n"})
    for e, g in N["ensemble"].items():
        print(e, {k: round(v, 1) for k, v in g.items()})
    for s, g in N["single"].items():
        print(s, {k: round(v, 1) for k, v in g.items()} if g else None)


if __name__ == "__main__":
    main()
