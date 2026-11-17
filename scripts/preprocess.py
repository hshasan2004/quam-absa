"""Convert EComReviews-BD and BABSA into one review-level JSONL format, with splits.

Output (data/processed/):
  ecomreviews.jsonl   one record per review
  babsa.jsonl         one record per unique text (duplicate rows merged)
  aspect_inventory.json   category -> [{"name", "query"}]
Report: reports/preprocess_report.txt

Record schema
  id, dataset ("eco" | "babsa"), domain, text, text_en (eco only, else null), script,
  overall ("positive" | "neutral" | "negative" | null),
  aspects: [{"name", "query", "label", "verbatim"}]
      eco:   every aspect of the category; label in {positive, negative, none}
      babsa: annotated aspects only;       label in {positive, neutral, negative}
  splits: {"<split name>": "train" | "val" | "test"}
      eco:   "official" (paper protocol), "grouped" (duplicates kept together),
             "zeroshot" (held-out categories)
      babsa: "grouped" (duplicates together, stratified by domain x sentiment)
"""
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.textutils import aspect_query, clean_text, dedup_key, script_of  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 42
META = {"Marketplace", "Category", "Product_Link", "Product_Name", "Brand", "Price_BDT",
        "Overall_Rating", "Reviews", "Sentiment"}
ECO_LABEL = {1: "positive", 0: "none", -1: "negative"}
ECO_OVERALL = {1: "positive", 0: "neutral", -1: "negative"}
# Zero-shot protocol: whole categories unseen in training. Chosen because most of their
# aspects (night view, firmware, wide angle, paper, ...) occur in no other category.
ZS_TEST = {"Router", "WebCam", "Mini_Printer", "CC_Camera"}
ZS_VAL = {"SmartWatch", "MemoryCards"}

log = io.StringIO()


def p(*a):
    print(*a, file=log)


# ------------------------------------------------------------------ helpers
def grouped_split(keys, strata, seed=SEED, frac=(0.70, 0.15, 0.15)):
    """Assign each item to train/val/test so that equal keys share a split.

    Groups are split per stratum (stratum of a group = stratum of its first item),
    filling val and test to their target fraction of rows, the rest going to train.
    """
    groups, g_stratum = defaultdict(list), {}
    for i, (k, s) in enumerate(zip(keys, strata)):
        groups[k].append(i)
        g_stratum.setdefault(k, s)
    by_stratum = defaultdict(list)
    for k, s in g_stratum.items():
        by_stratum[s].append(k)
    rng = random.Random(seed)
    out = [None] * len(keys)
    for s in sorted(by_stratum, key=str):
        ks = sorted(by_stratum[s])
        rng.shuffle(ks)
        n = sum(len(groups[k]) for k in ks)
        target = {"test": frac[2] * n, "val": frac[1] * n}
        filled = {"test": 0, "val": 0}
        for k in ks:
            size = len(groups[k])
            dest = "train"
            for part in ("test", "val"):
                if filled[part] + size / 2 <= target[part]:
                    dest = part
                    break
            if dest != "train":
                filled[dest] += size
            for i in groups[k]:
                out[i] = dest
    return out


def split_table(records, name):
    c = Counter((r["splits"][name]) for r in records)
    return " ".join(f"{k}={c[k]}" for k in ("train", "val", "test"))


# ------------------------------------------------------------------ EComReviews
def load_sheets(fname):
    frames = []
    for sheet, df in pd.read_excel(RAW / "ecomreviews" / fname, sheet_name=None).items():
        df = df.rename(columns=lambda c: str(c).strip())
        df.insert(0, "sheet", sheet)
        frames.append(df)
    return frames


def build_eco():
    p("#" * 70 + "\n# EComReviews-BD\n" + "#" * 70)
    main, eng = load_sheets("Product_Reviews_Main.xlsx"), load_sheets("Product_Reviews_ENGLISH.xlsx")
    inventory, records = {}, []
    pairs_kept = pairs_dropped = 0
    for mdf, edf in zip(main, eng):
        sheet = mdf["sheet"].iat[0]
        aspects = [c for c in mdf.columns if c not in META and c != "sheet"]
        inventory[sheet] = [{"name": a, "query": aspect_query(a)} for a in aspects]
        label_cols = [c for c in aspects + ["Sentiment"] if c in edf.columns]
        if len(label_cols) <= len(aspects):
            p(f"  {sheet}: English file lacks columns {sorted(set(aspects) - set(edf.columns))}")

        # English alignment: rows correspond by position until the first label mismatch.
        n = min(len(mdf), len(edf))
        same = (mdf[label_cols].iloc[:n].values == edf[label_cols].iloc[:n].values).all(axis=1)
        aligned = int(np.argmin(same)) if not same.all() else n
        pairs_kept += aligned
        pairs_dropped += len(mdf) - aligned
        if aligned < len(mdf):
            p(f"  {sheet}: English aligned for first {aligned}/{len(mdf)} rows")

        for i in range(len(mdf)):
            row = mdf.iloc[i]
            text = clean_text(row["Reviews"])
            records.append({
                "id": f"eco-{sheet}-{i}",
                "dataset": "eco",
                "domain": sheet,
                "text": text,
                "text_en": clean_text(edf["Reviews"].iat[i]) if i < aligned else None,
                "script": script_of(text),
                "overall": ECO_OVERALL[int(row["Sentiment"])],
                "aspects": [{"name": a, "query": aspect_query(a), "label": ECO_LABEL[int(row[a])],
                             "verbatim": None} for a in aspects],
                "product": clean_text(row["Product_Name"]),
                "splits": {},
            })
    p(f"Reviews: {len(records)} | English pairs aligned: {pairs_kept} | not aligned: {pairs_dropped}")

    # Official protocol from the dataset paper: 70/15/15 with train_test_split(random_state=42).
    # The paper does not say how sheets were concatenated or whether it stratified; we use
    # sheet order, no stratification. Sizes match the paper (14,479 / 3,103 / 3,103).
    idx = np.arange(len(records))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    for part, ids in (("train", tr), ("val", va), ("test", te)):
        for i in ids:
            records[i]["splits"]["official"] = part

    for r, s in zip(records, grouped_split([dedup_key(r["text"]) for r in records],
                                           [r["domain"] for r in records])):
        r["splits"]["grouped"] = s
    for r in records:
        r["splits"]["zeroshot"] = ("test" if r["domain"] in ZS_TEST else
                                   "val" if r["domain"] in ZS_VAL else "train")

    # Leakage check for the official split: identical texts across train and test.
    tr_keys = {dedup_key(r["text"]) for r in records if r["splits"]["official"] == "train"}
    leak = sum(dedup_key(r["text"]) in tr_keys for r in records if r["splits"]["official"] == "test")
    p(f"Official split:  {split_table(records, 'official')} | test texts also in train: {leak}")
    tr_keys = {dedup_key(r["text"]) for r in records if r["splits"]["grouped"] == "train"}
    leak = sum(dedup_key(r["text"]) in tr_keys for r in records if r["splits"]["grouped"] == "test")
    p(f"Grouped split:   {split_table(records, 'grouped')} | test texts also in train: {leak}")
    p(f"Zero-shot split: {split_table(records, 'zeroshot')} | test={sorted(ZS_TEST)} val={sorted(ZS_VAL)}")
    seen = {a["name"] for r in records if r["splits"]["zeroshot"] == "train" for a in r["aspects"]}
    unseen = sorted({a["name"] for c in ZS_TEST for a in inventory[c]} - seen)
    p(f"  zero-shot test aspects never seen in training ({len(unseen)}): {', '.join(unseen)}")

    empty = sum(not r["text"] for r in records)
    p(f"Empty review texts: {empty} (kept so split sizes match the paper; skipped at training time)")
    return records, inventory


# ------------------------------------------------------------------ BABSA
def norm_sentiment(s):
    s = s.strip().lower().split(":")[0].strip()
    if s.startswith("pos"):
        return "positive"
    if s.startswith(("neg", "ng")):
        return "negative"
    if s.startswith(("neu", "nue")):
        return "neutral"
    return None


def build_babsa():
    p("\n" + "#" * 70 + "\n# BABSA\n" + "#" * 70)
    df = pd.read_csv(RAW / "babsa_mendeley" / "final_set.csv")
    p(f"Rows in final_set.csv: {len(df)}")
    stats = Counter()
    merged = {}  # text -> {"domain", "pairs": {aspect: set(labels)}, "rows"}
    for text, asp, sen, dom in df[["text_content", "AnnotatedAspect", "AnnotatedSentiment",
                                   "MacroCategory"]].itertuples(index=False):
        text = clean_text(text)
        aspects = [clean_text(a) for a in str(asp).split(",") if clean_text(a)] if pd.notna(asp) else []
        sents = [norm_sentiment(s) for s in str(sen).split(",") if s.strip()] if pd.notna(sen) else []
        if not text or not aspects:
            stats["dropped: empty text or aspect"] += 1
            continue
        if len(sents) == 1 and len(aspects) > 1:
            sents = sents * len(aspects)
            stats["sentiment broadcast to all aspects"] += 1
        if len(sents) != len(aspects) or None in sents:
            stats["dropped: aspect/sentiment count mismatch or unknown label"] += 1
            continue
        m = merged.setdefault(text, {"domain": clean_text(dom), "pairs": defaultdict(set), "rows": 0})
        m["rows"] += 1
        for a, s in zip(aspects, sents):
            m["pairs"][a].add(s)

    records, conflicts, verb = [], 0, Counter()
    for i, (text, m) in enumerate(merged.items()):
        aspects = []
        for a, labels in m["pairs"].items():
            if len(labels) > 1:
                conflicts += 1
                continue
            v = a in text
            verb[v] += 1
            aspects.append({"name": a, "query": a, "label": labels.pop(), "verbatim": v})
        if not aspects:
            continue
        records.append({
            "id": f"babsa-{i}", "dataset": "babsa", "domain": m["domain"], "text": text,
            "text_en": None, "script": script_of(text), "overall": None,
            "aspects": aspects, "splits": {},
        })
    for k, v in stats.items():
        p(f"  {k}: {v}")
    p(f"Unique texts after merging duplicate rows: {len(records)} "
      f"(rows merged: {sum(m['rows'] for m in merged.values()) - len(merged)})")
    p(f"Aspect pairs dropped for conflicting labels on the same text: {conflicts}")
    p(f"Aspect pairs: {sum(verb.values())} | verbatim in text: {verb[True]} | not verbatim: {verb[False]}")
    p("Labels:", dict(Counter(a["label"] for r in records for a in r["aspects"])))

    strata = [f"{r['domain']}|{r['aspects'][0]['label']}" for r in records]
    for r, s in zip(records, grouped_split([dedup_key(r["text"]) for r in records], strata)):
        r["splits"]["grouped"] = s
    p(f"Grouped split: {split_table(records, 'grouped')}")
    test_domains = Counter(r["domain"] for r in records if r["splits"]["grouped"] == "test")
    p("Test-set domains:", dict(test_domains.most_common()))
    return records


# ------------------------------------------------------------------ main
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eco, inventory = build_eco()
    babsa = build_babsa()
    write_jsonl(OUT / "ecomreviews.jsonl", eco)
    write_jsonl(OUT / "babsa.jsonl", babsa)
    (OUT / "aspect_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    p(f"\nWrote {OUT / 'ecomreviews.jsonl'}, {OUT / 'babsa.jsonl'}, {OUT / 'aspect_inventory.json'}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "preprocess_report.txt").write_text(log.getvalue(), encoding="utf-8")
    print(log.getvalue())


if __name__ == "__main__":
    main()
