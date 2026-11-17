"""Collect runs/*/metrics.json into reports/results.md."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
PAPER = [("XLM-R (paper)", 27.26, 80.92, 40.78), ("mBERT (paper)", 29.81, 74.68, 42.61),
         ("BanglaBERT (paper)", 31.77, 77.13, 45.00)]


def pct(x):
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) else "–"


def main():
    runs = {}
    for f in sorted((ROOT / "runs").glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        cfg = json.loads((f.parent / "config.json").read_text(encoding="utf-8"))
        runs[f.parent.name] = (m, cfg)

    lines = ["# Results (test set, best validation epoch)\n"]
    eco = {k: v for k, v in runs.items() if "eco" in v[0]["test"] and not k.startswith("overfit")}
    if eco:
        lines += ["## EComReviews-BD\n",
                  "| run | split | layout | readout | presence | pair P | pair R | **pair F1** | det F1 | det macro-F1 | sent acc | overall F1 | epoch |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, p_, r_, f_ in PAPER:
            lines.append(f"| {name} | official | – | – | – | {p_:.1f} | {r_:.1f} | **{f_:.1f}** | – | – | – | – | – |")
        for k, (m, c) in eco.items():
            e = m["test"]["eco"]
            lines.append(f"| {k} | {c['eco_split']} | {c['layout']} | {c['readout']} | {c['presence_mode']}"
                         f"{'' if not c['no_domain'] else ', no-domain'}{'' if not c['cons_weight'] else ', cons'} | "
                         f"{pct(e['pair_P'])} | {pct(e['pair_R'])} | **{pct(e['pair_F1'])}** | {pct(e['det_F1'])} | "
                         f"{pct(e['det_macroF1'])} | {pct(e['sent_acc_on_gold'])} | {pct(e['overall_macroF1'])} | {m['best_epoch']} |")
    bab = {k: v for k, v in runs.items() if "babsa" in v[0]["test"]}
    if bab:
        lines += ["\n## BABSA (grouped split)\n",
                  "| run | sent acc | **sent macro-F1** | verbatim F1 | non-verbatim F1 | presence AUROC |",
                  "|---|---|---|---|---|---|"]
        for k, (m, c) in bab.items():
            b = m["test"]["babsa"]
            lines.append(f"| {k} | {pct(b['sent_acc'])} | **{pct(b['sent_macroF1'])}** | {pct(b.get('sent_macroF1_verbatim'))} | "
                         f"{pct(b.get('sent_macroF1_nonverbatim'))} | {pct(b.get('presence_AUROC'))} |")
    out = "\n".join(lines) + "\n"
    (ROOT / "reports" / "results.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
