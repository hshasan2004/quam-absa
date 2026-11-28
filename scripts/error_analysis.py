"""Per-unseen-aspect comparison and examples: rag3 vs no retrieval (3-seed ensembles, zero-shot test).

    python scripts/error_analysis.py          -> reports/error_analysis.json (+ printed tables)
"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from quam.data import load_records  # noqa: E402
from ensemble_eval import GROUPS, ensemble, seen  # noqa: E402
from build_rag_kb import TOK, pkey  # noqa: E402

recs = {r["id"]: r for r in load_records(("eco",))}
kb = {k: [x["form"] for x in v[:4]] for k, v in json.loads((ROOT / "data/rag2_kb_zeroshot.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
A, p_none, s_none = ensemble(GROUPS["zeroshot"]["no retrieval"])
_, p_rag, s_rag = ensemble(GROUPS["zeroshot"]["rag3"])
gold = A["gold_pres"] == 1
ok_none = p_none & gold & (s_none == A["gold_sent"]); ok_rag = p_rag & gold & (s_rag == A["gold_sent"])
f1 = lambda tp, p, g: 200 * tp / max(p + g, 1)
rows = []
for n in sorted({x for x in A["name"] if x not in seen}):
    m = A["name"] == n
    if gold[m].sum() == 0:
        continue
    rows.append({"aspect": n, "gold": int(gold[m].sum()), "forms": kb.get(n, []),
                 "none_pairF1": f1((ok_none & m).sum(), (p_none & m).sum(), gold[m].sum()),
                 "rag3_pairF1": f1((ok_rag & m).sum(), (p_rag & m).sum(), gold[m].sum())})
rows.sort(key=lambda r: r["rag3_pairF1"] - r["none_pairF1"], reverse=True)
print(f"{'unseen aspect':16s} {'gold':>5s} {'none':>6s} {'rag3':>6s} {'diff':>6s}  retrieved forms")
for r in rows:
    print(f"{r['aspect']:16s} {r['gold']:5d} {r['none_pairF1']:6.1f} {r['rag3_pairF1']:6.1f} {r['rag3_pairF1'] - r['none_pairF1']:+6.1f}  {' ; '.join(r['forms'])}")
won = np.array([x not in seen for x in A["name"]]) & ok_rag & ~p_none
lost = np.array([x not in seen for x in A["name"]]) & ok_none & ~p_rag
print(f"\nunseen gold mentions found only by rag3: {int(won.sum())}, only by no retrieval: {int(lost.sum())}")
ex = []
for i in np.where(won)[0]:
    r = recs[A["id"][i]]; words = {w.lower() for w in TOK.findall(r["text"])}; keys = {pkey(w) for w in words}
    hit = [f for f in kb.get(A["name"][i], []) if f.lower() in words or (len(pkey(f)) >= 3 and pkey(f) in keys)]
    if hit and len(r["text"]) < 260:
        ex.append({"aspect": A["name"][i], "category": r["domain"], "matched_form": hit[0], "review": r["text"],
                   "polarity": ["positive", "neutral", "negative"][int(A["gold_sent"][i])] if A["gold_sent"][i] in (0, 1, 2) else int(A["gold_sent"][i])})
seen_a = set(); out_ex = []
for e in ex:
    if e["aspect"] not in seen_a:
        seen_a.add(e["aspect"]); out_ex.append(e)
print("\nexamples (unseen aspect found by rag3, missed by no retrieval; a retrieved form occurs in the review):")
for e in out_ex[:8]:
    print(f"- [{e['category']}] {e['aspect']} via '{e['matched_form']}' ({e['polarity']}): {e['review']}")
(ROOT / "reports" / "error_analysis.json").write_text(json.dumps({"per_aspect": rows, "won": int(won.sum()), "lost": int(lost.sum()),
                                                                   "examples": out_ex}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
