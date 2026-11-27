"""Intrinsic retrieval quality on SEEN aspects only (no test-set information is used).

    python scripts/eval_retrieval.py data/rag2_kb_zeroshot.json [more KBs ...]

Reference = the hand-written multi-script KB (data/aspect_kb.json), restricted to forms that occur in
the candidate vocabulary of the zero-shot TRAINING reviews (a retriever cannot find anything else).
A retrieved form counts as correct when it equals a reference form or has the same cross-script
phonetic key (e.g. 'ডেলিভারি' vs 'ডেলিভারী'). Reports precision, recall (of findable reference forms),
the share of seen aspects with at least one correct form, and the share whose forms are all wrong.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from build_rag_kb import candidates, pkey  # noqa: E402
from quam.data import load_records  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
recs = load_records(("eco",))
seen = sorted({a["query"] for r in recs if r["splits"]["zeroshot"] == "train" for a in r["aspects"] if a["label"] != "none"})
vocab = set(candidates([r["text"] for r in recs if r["splits"]["zeroshot"] == "train"], 3))
hand = {k: [f for fl in v.values() for f in fl] for k, v in json.loads((ROOT / "data/aspect_kb.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}


def match(f, refs):
    return f in refs or pkey(f) in {pkey(r) for r in refs if len(pkey(r)) >= 2}


for path in sys.argv[1:]:
    kb = {k: v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    tp = n_ret = n_ref = found = 0; any_ok = all_bad = 0
    for a in seen:
        refs = [f.lower() for f in hand.get(a, []) if f.lower() in vocab]
        ret = [x["form"] if isinstance(x, dict) else x for x in kb.get(a, [])]
        ok = [match(f, refs) for f in ret]
        tp += sum(ok); n_ret += len(ret); n_ref += len(refs)
        found += sum(any(match(f, [r]) for f in ret) for r in refs)
        any_ok += any(ok); all_bad += bool(ret) and not any(ok)
    print(f"{Path(path).name:28s} precision {100 * tp / max(n_ret, 1):5.1f}  recall {100 * found / max(n_ref, 1):5.1f}  "
          f"aspects with a correct form {100 * any_ok / len(seen):5.1f}%  all-wrong {100 * all_bad / len(seen):5.1f}%  "
          f"(forms/aspect {n_ret / len(seen):.1f}, {len(seen)} seen aspects)")
