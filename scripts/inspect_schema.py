"""Print the raw structure of both datasets (columns, dtypes, sample values)."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pd.set_option("display.width", 200, "display.max_columns", 200, "display.max_colwidth", 80)


def show(name, df):
    print(f"\n{'=' * 80}\n{name}: {df.shape[0]} rows x {df.shape[1]} cols\n{'=' * 80}")
    for c in df.columns:
        s = df[c]
        top = s.dropna().astype(str).value_counts().head(5).to_dict()
        print(f"- {c!r:40} dtype={str(s.dtype):8} null={s.isna().sum():6} nunique={s.nunique():6} top={top}")


eco_main = pd.read_excel(RAW / "ecomreviews" / "Product_Reviews_Main.xlsx", sheet_name=None)
for sheet, df in eco_main.items():
    show(f"EComReviews MAIN [{sheet}]", df)

eco_en = pd.read_excel(RAW / "ecomreviews" / "Product_Reviews_ENGLISH.xlsx", sheet_name=None)
for sheet, df in eco_en.items():
    show(f"EComReviews ENGLISH [{sheet}]", df)

for f in ["final_set.csv", "total.csv"]:
    show(f"BABSA mendeley/{f}", pd.read_csv(RAW / "babsa_mendeley" / f))

show("BABSA github/test_set.csv", pd.read_csv(RAW / "babsa_repo" / "data" / "main" / "test_set.csv"))
for sheet, df in pd.read_excel(RAW / "babsa_repo" / "data" / "main" / "main.xlsx", sheet_name=None).items():
    show(f"BABSA github/main.xlsx [{sheet}]", df)
