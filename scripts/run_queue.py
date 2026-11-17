"""Run the main experiments one after another. Finished runs (metrics.json exists) are skipped,
so the queue can be restarted safely. Progress: runs/queue_log.txt
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# Main model: mLSTM-only ("mm"). The sLSTM variant ("mmsm") scored lower (official pair F1
# 39.7 vs 52.2) and was 4x slower; its finished runs eco_official_full / eco_grouped_full
# are kept as the "with sLSTM" ablation rows.
FULL = ["--layout", "mm"]
G = ["--data", "eco", "--eco-split", "grouped", *FULL]

RUNS = [
    # name                    extra arguments
    ("eco_official_mm",       ["--data", "eco", "--eco-split", "official", *FULL]),   # already done
    ("eco_grouped_mm",        G),
    # core ablation: query the memory vs mean-pool the tokens (AF-MAT style)
    # (presence from the pooled value too, so this run uses no memory read at all)
    ("abl_meanpool",          [*G, "--readout", "meanpool", "--presence-mode", "value"]),
    # where does presence come from: memory magnitude only vs retrieved value only
    ("abl_pres_mag",          [*G, "--presence-mode", "mag"]),
    ("abl_pres_val",          [*G, "--presence-mode", "value"]),
    # remaining components
    ("abl_nodomain",          [*G, "--no-domain"]),
    ("abl_depth3",            [*G[:-2], "--layout", "mmm"]),
    ("eco_grouped_cons",      [*G, "--cons-weight", "0.1"]),
    ("eco_zeroshot_mm",       ["--data", "eco", "--eco-split", "zeroshot", *FULL]),
    ("babsa_mm",              ["--data", "babsa", *FULL]),
]
# Extra seeds (seed 42 is the run above) for the comparisons the paper depends on.
# summarize.py groups "<name>_s<seed>" with "<name>" and reports mean ± std.
SEEDED = {"eco_grouped_mm": G,
          "abl_meanpool": [*G, "--readout", "meanpool", "--presence-mode", "value"],
          "abl_pres_val": [*G, "--presence-mode", "value"]}
for _seed in (1, 2):
    for _name, _args in SEEDED.items():
        RUNS.append((f"{_name}_s{_seed}", [*_args, "--seed", str(_seed)]))


def main():
    only = set(sys.argv[1:])
    log = open(ROOT / "runs" / "queue_log.txt", "a", encoding="utf-8")
    for name, extra in RUNS:
        if only and name not in only:
            continue
        if (ROOT / "runs" / name / "metrics.json").exists():
            print(f"skip {name} (done)", file=log, flush=True)
            continue
        t0 = time.time()
        print(f"{time.strftime('%H:%M')} start {name}", file=log, flush=True)
        rc = subprocess.call([PY, str(ROOT / "scripts" / "train.py"), "--name", name, *extra],
                             cwd=ROOT, stdout=subprocess.DEVNULL, stderr=open(ROOT / "runs" / f"{name}.err", "w"))
        print(f"{time.strftime('%H:%M')} end   {name} rc={rc} {(time.time() - t0) / 60:.1f} min", file=log, flush=True)


if __name__ == "__main__":
    main()
