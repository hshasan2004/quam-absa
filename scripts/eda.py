"""Exploratory analysis of EComReviews-BD and BABSA.

Answers the questions that decide the QuAM-xLSTM design:
  * label formats, class balance, aspects per review
  * meaning of EComReviews aspect value 0 (neutral vs not mentioned)
  * whether the English file is row-aligned with the main file
  * whether Overall_Rating is per-review or per-product
  * script / language mix, text lengths, duplicates
  * whether BABSA aspects occur verbatim in the text (needed for span tagging)
  * whether the GitHub BABSA test_set.csv overlaps final_set.csv
Writes the report to reports/eda_report.txt and prints it.
"""
import io
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ["Marketplace", "Category", "Product_Link", "Product_Name", "Brand", "Price_BDT",
        "Overall_Rating", "Reviews", "Sentiment"]
out = io.StringIO()


def p(*a):
    print(*a, file=out)


def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)" if d else str(n)


BN = re.compile(r"[ঀ-৿]")
LAT = re.compile(r"[A-Za-z]")
# Common romanised-Bangla tokens; heuristic only, used to split Latin-script text.
BANGLISH = set("""valo bhalo valo valoi kharap kharapi onek anek ase ache achhe nai nei na hoy hoyeche hoise
hoyse dise diyeche dilam pelam paisi paichi peyechi ekdom amar ami apni apnar tader taka dam bhai vai vaia
kintu tobe ar r jonno moto mone kore korse korche korlam chilo silo khub khubi sob shob kono kichu ekta
eta oita ota product ta ti gulo gula thik shundor sundor sondor dorkar lagbe lage lagse laglo deoya deya
oneek valoo bad na""".split()) - {"product", "bad"}


def script_of(text):
    t = str(text)
    bn, lat = len(BN.findall(t)), len(LAT.findall(t))
    if bn + lat == 0:
        return "other"
    if bn / (bn + lat) >= 0.8:
        return "bangla"
    if lat / (bn + lat) >= 0.8:
        toks = re.findall(r"[a-z]+", t.lower())
        hits = sum(tok in BANGLISH for tok in toks)
        return "banglish" if toks and hits / len(toks) >= 0.15 else "english"
    return "mixed"


def words(text):
    return len(str(text).split())


def length_line(series):
    q = series.quantile([0.5, 0.9, 0.95, 0.99]).astype(int).tolist()
    return f"median={q[0]} p90={q[1]} p95={q[2]} p99={q[3]} max={series.max()}"


# ---------------------------------------------------------------- EComReviews
def load_eco(fname):
    sheets = pd.read_excel(RAW / "ecomreviews" / fname, sheet_name=None)
    frames = []
    for name, df in sheets.items():
        df = df.rename(columns=lambda c: str(c).strip())
        df["sheet"] = name
        df["row_in_sheet"] = range(len(df))
        frames.append(df)
    return sheets, frames


p("#" * 80, "\n# EComReviews-BD\n" + "#" * 80)
main_sheets, main_frames = load_eco("Product_Reviews_Main.xlsx")
en_sheets, en_frames = load_eco("Product_Reviews_ENGLISH.xlsx")

rows, all_aspects, long_rows = 0, Counter(), []
p(f"\n{'sheet':16} {'rows':>5}  aspects")
for df in main_frames:
    aspects = [c for c in df.columns if c not in META + ["sheet", "row_in_sheet"]]
    all_aspects.update(aspects)
    p(f"{df['sheet'].iat[0]:16} {len(df):5}  {len(aspects):2}: {', '.join(aspects)}")
    rows += len(df)
    a = df[aspects].apply(pd.to_numeric, errors="coerce")
    for i in range(len(df)):
        vals = a.iloc[i]
        long_rows.append({
            "sheet": df["sheet"].iat[0], "review": df["Reviews"].iat[i],
            "sentiment": df["Sentiment"].iat[i], "product": df["Product_Name"].iat[i],
            "rating": df["Overall_Rating"].iat[i],
            "n_pos": int((vals == 1).sum()), "n_neg": int((vals == -1).sum()),
            "n_zero": int((vals == 0).sum()), "n_nan": int(vals.isna().sum()),
            "n_other": int((~vals.isin([1, 0, -1]) & vals.notna()).sum()),
            "n_aspects": len(aspects),
        })
eco = pd.DataFrame(long_rows)
p(f"\nTotal reviews: {rows}   sheets: {len(main_frames)}   unique aspect names: {len(all_aspects)}")
p("Aspects shared by most sheets:", ", ".join(f"{k}({v})" for k, v in all_aspects.most_common(8)))
p("Aspects unique to one sheet:", sum(v == 1 for v in all_aspects.values()))

p("\nOverall sentiment:", eco["sentiment"].value_counts().to_dict())
cells = eco[["n_pos", "n_neg", "n_zero", "n_nan", "n_other"]].sum()
p("Aspect cells:", {k: int(v) for k, v in cells.items()}, f"of {int(eco['n_aspects'].sum())}")
eco["n_mentioned"] = eco["n_pos"] + eco["n_neg"]
p("Non-zero aspects per review:", eco["n_mentioned"].value_counts().sort_index().to_dict())
p("Reviews with >=1 non-zero aspect:", pct(int((eco["n_mentioned"] > 0).sum()), len(eco)))
p("Reviews with >=2 non-zero aspects:", pct(int((eco["n_mentioned"] > 1).sum()), len(eco)))

p("\n-- Meaning of aspect value 0 --")
p("If 0 also meant 'neutral', neutral-sentiment reviews would have no +-1 aspects more often.")
p(pd.crosstab(eco["sentiment"], eco["n_mentioned"].clip(upper=3), margins=True).to_string())
both = eco[(eco["n_pos"] > 0) & (eco["n_neg"] > 0)]
p("Reviews with both a +1 and a -1 aspect (mixed):", pct(len(both), len(eco)))
p("Overall sentiment of mixed reviews:", both["sentiment"].value_counts().to_dict())

p("\n-- Overall_Rating: per review or per product? --")
per_prod = eco.groupby(["sheet", "product"])["rating"].nunique()
p("Products:", len(per_prod), "| products with >1 distinct rating value:", int((per_prod > 1).sum()))
rnum = pd.to_numeric(eco["rating"], errors="coerce")
p("Distinct rating values overall:", sorted(rnum.dropna().unique().tolist()),
  "| non-numeric:", eco.loc[rnum.isna() & eco["rating"].notna(), "rating"].astype(str).value_counts().to_dict())

eco["script"] = eco["review"].map(script_of)
eco["words"] = eco["review"].map(words)
p("\nScript (heuristic):", eco["script"].value_counts().to_dict())
p("Review length (words):", length_line(eco["words"]))
dup_txt = eco["review"].astype(str).str.strip().str.lower()
p("Duplicate review texts:", pct(int(dup_txt.duplicated().sum()), len(eco)))
conflict = eco.assign(t=dup_txt).groupby("t")["sentiment"].nunique()
p("Duplicate texts with conflicting overall sentiment:", int((conflict > 1).sum()))
p("Most common texts:", dup_txt.value_counts().head(8).to_dict())

p("\n-- ENGLISH file alignment --")
p(f"{'main sheet':16} {'en sheet':16} rows_main rows_en labels_equal text_equal")
total_eq_lab = total_eq_txt = total_rows = 0
for mdf, edf in zip(main_frames, en_frames):
    n = min(len(mdf), len(edf))
    label_cols = [c for c in mdf.columns if c not in META + ["sheet", "row_in_sheet"]] + ["Sentiment"]
    common = [c for c in label_cols if c in edf.columns]
    lab_eq = int((mdf[common].iloc[:n].values == edf[common].iloc[:n].values).all(axis=1).sum())
    txt_eq = int((mdf["Reviews"].astype(str).iloc[:n].values == edf["Reviews"].astype(str).iloc[:n].values).sum())
    total_eq_lab += lab_eq; total_eq_txt += txt_eq; total_rows += n
    p(f"{mdf['sheet'].iat[0]:16} {edf['sheet'].iat[0]:16} {len(mdf):9} {len(edf):7} {lab_eq:12} {txt_eq:10}")
p(f"TOTAL rows compared {total_rows}: labels identical {pct(total_eq_lab, total_rows)}, "
  f"text identical {pct(total_eq_txt, total_rows)}")
m0 = pd.concat(main_frames, ignore_index=True)
e0 = pd.concat(en_frames, ignore_index=True)
if len(m0) == len(e0):
    m0["script"] = m0["Reviews"].map(script_of)
    changed = m0["Reviews"].astype(str).values != e0["Reviews"].astype(str).values
    p("Rows whose text differs, by original script:", m0.loc[changed, "script"].value_counts().to_dict())
    p("Example pairs (original -> English):")
    for i in m0.index[changed][:: max(1, changed.sum() // 6)][:6]:
        p(f"  [{m0.at[i, 'script']}] {str(m0.at[i, 'Reviews'])[:90]!r}\n      -> {str(e0.at[i, 'Reviews'])[:90]!r}")

# ---------------------------------------------------------------- BABSA
p("\n" + "#" * 80, "\n# BABSA\n" + "#" * 80)
b = pd.read_csv(RAW / "babsa_mendeley" / "final_set.csv")
p("final_set.csv rows:", len(b), "| columns:", list(b.columns))


def split_list(x):
    return [s.strip() for s in str(x).split(",") if s.strip()] if pd.notna(x) else []


b["aspects"] = b["AnnotatedAspect"].map(split_list)
b["sents"] = b["AnnotatedSentiment"].map(lambda x: [s.lower() for s in split_list(x)])
b["n_asp"], b["n_sent"] = b["aspects"].map(len), b["sents"].map(len)
p("Aspects per row:", b["n_asp"].value_counts().sort_index().to_dict())
p("Rows where #aspects != #sentiments:", pct(int((b["n_asp"] != b["n_sent"]).sum()), len(b)))
p("Sentiment labels (per aspect):", dict(Counter(s for ss in b["sents"] for s in ss).most_common()))
p("MacroCategory:", b["MacroCategory"].value_counts().to_dict())

pairs = [(t, a) for t, asp in zip(b["text_content"].astype(str), b["aspects"]) for a in asp]
verb = sum(a in t for t, a in pairs)
verb_ci = sum(a.lower() in t.lower() for t, a in pairs)
p("Aspect terms found verbatim in text:", pct(verb, len(pairs)), "| case-insensitive:", pct(verb_ci, len(pairs)))
miss = [(t, a) for t, a in pairs if a not in t]
p("Examples NOT verbatim:")
for t, a in miss[:: max(1, len(miss) // 6)][:6]:
    p(f"  aspect={a!r}  text={t[:90]!r}")

b["script"] = b["text_content"].map(script_of)
b["words"] = b["text_content"].map(words)
p("\nScript (heuristic):", b["script"].value_counts().to_dict())
p("Text length (words):", length_line(b["words"]))
bt = b["text_content"].astype(str).str.strip()
p("Duplicate texts:", pct(int(bt.duplicated().sum()), len(b)))
p("Duplicate (text, aspect) with conflicting sentiment:",
  int(b.assign(t=bt).groupby(["t", "AnnotatedAspect"])["AnnotatedSentiment"].nunique().gt(1).sum()))

p("\nSentiment by MacroCategory (row-level first label):")
b["first_sent"] = b["sents"].map(lambda s: s[0] if s else None)
p(pd.crosstab(b["MacroCategory"], b["first_sent"], margins=True).to_string())

t = pd.read_csv(RAW / "babsa_repo" / "data" / "main" / "test_set.csv")
ts = set(t["review"].astype(str).str.strip())
p(f"\nGitHub test_set.csv rows: {len(t)} | texts also in final_set: {pct(len(ts & set(bt)), len(ts))}")
p("test_set sentiment:", t["annotated_sentiment"].value_counts().to_dict())
p("test_set script:", t["review"].map(script_of).value_counts().to_dict())

report = out.getvalue()
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "eda_report.txt").write_text(report, encoding="utf-8")
print(report)
