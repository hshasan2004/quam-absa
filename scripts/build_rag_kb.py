"""Automatic aspect-knowledge retrieval (RAMP-R): no hand-written synonym lists.

    python scripts/build_rag_kb.py --split zeroshot      -> data/rag_kb_zeroshot.json

For every aspect name (seen or unseen), candidate surface forms are retrieved from the words and
bigrams of the TRAINING reviews of the chosen split only (no test text), by two retrievers:

  phonetic  every string (English, Banglish, Bangla script) is mapped to one cross-script
            phonetic key (Bangla letters -> Latin sounds, Latin spelling rules unified, vowels
            dropped). Banglish 'renj', Bangla 'রেঞ্জ' and English 'range' all become 'rnj'.
            Score = normalized edit similarity of the keys.
  dense     LaBSE sentence embeddings (multilingual, language-agnostic): cosine(name, candidate),
            finds synonyms by meaning ('range' -> 'coverage', 'signal').

The two rankings are fused with reciprocal-rank fusion; the top forms (at most --per-script per
script, --k in total) are written with their fused score, which the model's evidence gate uses.
Output: {aspect: [{"form": str, "score": float, "src": "phon"|"dense"|"both", "script": "bn"|"lat"}]}
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import load_records  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- cross-script phonetic key
BN = {**dict.fromkeys("কখ", "k"), **dict.fromkeys("গঘ", "g"), "ঙ": "ng", **dict.fromkeys("চছ", "c"),
      **dict.fromkeys("জঝয", "j"), "ঞ": "n", **dict.fromkeys("টঠতথৎ", "t"), **dict.fromkeys("ডঢদধ", "d"),
      **dict.fromkeys("ণন", "n"), "প": "p", "ফ": "f", **dict.fromkeys("বভ", "b"), "ম": "m", "র": "r",
      "ড়": "r", "ঢ়": "r", "ল": "l", **dict.fromkeys("শষস", "s"), "হ": "h", "য়": "y", "ং": "ng",
      "ঃ": "h", "ঁ": "", "অ": "o", "আ": "a", "া": "a", **dict.fromkeys("ইঈিী", "i"), **dict.fromkeys("উঊুূ", "u"),
      "ঋ": "ri", "ৃ": "ri", "এ": "e", "ে": "e", "ঐ": "oi", "ৈ": "oi", "ও": "o", "ো": "o", "ঔ": "ou", "ৌ": "ou"}
BN_RE = re.compile(r"[ঀ-৿]")
LAT_RULES = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("bh", "b"), ("th", "t"), ("dh", "d"), ("kh", "k"),
             ("gh", "g"), ("ch", "c"), ("sh", "s"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("x", "ks"),
             ("z", "j"), ("v", "b"), ("w", "u")]
VOWELS = re.compile(r"[aeiouy]")


def to_latin(s):
    s = s.lower()
    if BN_RE.search(s):
        s = s.replace("ড়", "ড়").replace("ঢ়", "ঢ়").replace("য়", "য়")
        s = re.sub("্[যব]", "", s)                # ya-/ba-phala: vowel colouring, not a sound
        s = "".join(BN.get(ch, "" if BN_RE.match(ch) else ch) for ch in s)
    return s


def pkey(s):
    """Cross-script phonetic key: consonant skeleton with unified spelling rules."""
    s = re.sub(r"[^a-z ]", "", to_latin(s))
    s = re.sub(r"c(?=[eiy])", "s", s)                       # price -> prise
    s = re.sub(r"g(?=[ei])", "j", s)                        # range -> ranje
    for a, b in LAT_RULES:
        s = s.replace(a, b)
    s = s.replace("c", "k").replace(" ", "")
    s = VOWELS.sub("", s)
    return re.sub(r"(.)\1+", r"\1", s)                      # collapse doubles: battery -> btr


# ---------------------------------------------------------------- candidates
TOK = re.compile(r"[ঀ-৿]+|[A-Za-z]+")
STOP = set("""the a an is it this that and or but to of in on for with was are be very so not no its my i you
he she we they me product good bad nice ok all one just also from at as by have has had do did will can
ami amar ar o na ki ta te ei oi ekta khub onek valo vlo bhalo kharap hoyeche ache chilo kore korche dise
diye diyeche pai paisi peyechi kintu jonno moto""".split())


def candidates(texts, min_freq):
    c = Counter()
    for t in texts:
        w = [x.lower() for x in TOK.findall(t)]
        c.update(x for x in w if len(x) > 1 and x not in STOP)
        c.update(f"{a} {b}" for a, b in zip(w, w[1:]) if a not in STOP and b not in STOP)
    return [w for w, n in c.items() if n >= min_freq]


@torch.no_grad()
def labse(strings, bs=512):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    m = AutoModel.from_pretrained("sentence-transformers/LaBSE").cuda().eval().half()
    out = []
    for i in range(0, len(strings), bs):
        b = tok(strings[i:i + bs], padding=True, truncation=True, max_length=16, return_tensors="pt").to("cuda")
        out.append(torch.nn.functional.normalize(m(**b).pooler_output.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="zeroshot", choices=["official", "grouped", "zeroshot"])
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-script", type=int, default=4)
    ap.add_argument("--phon-min", type=float, default=0.75, help="min phonetic key similarity")
    ap.add_argument("--dense-min", type=float, default=0.55, help="min LaBSE cosine")
    ap.add_argument("--phon-dense-min", type=float, default=0.50, help="phonetic hits need this LaBSE cosine")
    ap.add_argument("--dense-strict", type=float, default=0.65, help="min LaBSE cosine for meaning-only forms")
    ap.add_argument("--margin", type=float, default=0.0, help="form must be (almost) closest to this aspect")
    ap.add_argument("--evid-min", type=int, default=3, help="aspect needs this many translations naming it to be verifiable")
    ap.add_argument("--verify-min", type=float, default=0.05, help="min share of a form's reviews whose translation names the aspect")
    ap.add_argument("--pivot-min", type=float, default=0.3, help="pivot retriever: min share")
    ap.add_argument("--pivot-dense-min", type=float, default=0.45, help="pivot-only forms need this LaBSE cosine")
    ap.add_argument("--pool", type=int, default=30, help="top-N per retriever before fusion")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = load_records(("eco",))
    names = sorted({a["query"] for r in recs for a in r["aspects"]})
    texts = [r["text"] for r in recs if r["splits"][args.split] == "train"]
    cand = candidates(texts, args.min_freq)
    print(f"{len(texts)} training reviews -> {len(cand)} candidate forms; {len(names)} aspect names")

    # phonetic retriever
    ck = [pkey(c) for c in cand]
    nk = [pkey(n) for n in names]
    P = cdist(nk, ck, scorer=Levenshtein.normalized_similarity, workers=-1)
    for j, k in enumerate(ck):
        if len(k) < 2:
            P[:, j] = 0
    # dense retriever
    E_c, E_n = labse(cand), labse(names)
    D = E_n @ E_c.T

    # translation-pivot evidence: the dataset gives an English translation of every review; a true form of
    # aspect a should occur in reviews whose translation names a (only TRAINING reviews are used)
    tr = [r for r in recs if r["splits"][args.split] == "train" and r.get("text_en")]
    inv = {}
    for d, r in enumerate(tr):
        w = [x.lower() for x in TOK.findall(r["text"])]
        for x in set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}:
            inv.setdefault(x, set()).add(d)
    en = [r["text_en"].lower() for r in tr]
    GENERIC = {"and", "quality", "view", "low", "wide"}

    def name_re(n):
        ws = [w for w in n.split() if w not in GENERIC] or n.split()
        pats = [(r"\b" + w + r"s?\b") if len(w) <= 3 else (r"\b" + (w[:4] if len(w) > 6 else w)) for w in ws]
        return re.compile("|".join(pats))
    hits = {n: {d for d, t in enumerate(en) if name_re(n).search(t)} for n in names}
    sup = np.array([len(inv.get(c, ())) for c in cand])

    # noise filters (precision over recall: an aspect may get 0 forms)
    #  exclusive: a form goes only to the aspect it is closest to (no other aspect's synonym / name)
    #  redundant: Latin inflections of the name itself ('range e', 'photos') add nothing
    top2_D = np.sort(D, 0)[-2:]                                    # best and second-best aspect per candidate
    top2_P = np.sort(P, 0)[-2:]
    name_set = set(names)
    kb = {}
    for i, n in enumerate(names):
        own = {n, n.replace(" ", "")}
        nwords = set(n.split())
        best_other_D = np.where(D[i] >= top2_D[1], top2_D[0], top2_D[1])
        best_other_P = np.where(P[i] >= top2_P[1], top2_P[0], top2_P[1])
        redundant = np.array([not BN_RE.search(c) and (ck[j].startswith(nk[i]) or bool(nwords & set(c.split())))
                              for j, c in enumerate(cand)])
        H = hits[n]
        evid = len(H) >= args.evid_min                              # can this aspect be verified at all?
        h = np.array([len(inv.get(c, set()) & H) for c in cand]) if evid else np.zeros(len(cand))
        prec = h / np.maximum(sup, 1)
        lists = {}
        if evid:                                                    # third retriever: pivot co-occurrence
            pv = (sup >= 3) & (h >= 2) & (prec >= args.pivot_min) & ~redundant
            pv &= (D[i] >= args.pivot_dense_min) & (D[i] >= best_other_D)   # co-occurring is not enough: related meaning, this aspect
            pv &= np.array([c not in name_set for c in cand])
            lists["pivot"] = [j for j in np.argsort(-(prec * np.log1p(sup))) if pv[j] and cand[j] not in own][: args.pool]
        for src, S, lo in (("phon", P[i], args.phon_min), ("dense", D[i], args.dense_min)):
            ok = (S >= lo) & ~redundant & (D[i] >= best_other_D - args.margin)
            if src == "phon":                                   # sound-alike must not contradict meaning
                ok &= ((D[i] >= args.phon_dense_min) | (S >= 0.95)) & (P[i] > best_other_P)
            order = [j for j in np.argsort(-S) if ok[j] and cand[j] not in own][: args.pool]
            lists[src] = order
        fused = Counter()
        for src, order in lists.items():
            for rank, j in enumerate(order):
                fused[j] += 1.0 / (10 + rank)                   # reciprocal-rank fusion
        out, per, seen_keys = [], Counter(), set()
        for j, s in fused.most_common():
            sc = "bn" if BN_RE.search(cand[j]) else "lat"
            key = (pkey(cand[j]), sc)
            if per[sc] >= args.per_script or key in seen_keys:
                continue
            srcs = [k for k, o in lists.items() if j in o]
            src = "+".join(srcs)
            verified = evid and sup[j] >= 2 and prec[j] >= args.verify_min
            if evid and sup[j] >= 3 and not verified:               # testable and failed the translation check
                continue
            if not verified and "pivot" not in srcs:                # untestable: need phonetic AND dense agreement
                if not ("phon" in srcs and "dense" in srcs) and not (srcs == ["dense"] and D[i, j] >= args.dense_strict):
                    continue
            out.append({"form": cand[j], "score": round(float(s) * 10, 3), "src": src, "script": sc,
                        "pivot": round(float(prec[j]), 3)})
            per[sc] += 1; seen_keys.add(key)
            if len(out) >= args.k:
                break
        kb[n] = out
    path = Path(args.out or ROOT / "data" / f"rag_kb_{args.split}.json")
    meta = {"_about": f"automatic retrieval from {args.split}-train reviews only; phonetic+LaBSE+translation-pivot, RRF", "_args": vars(args)}
    path.write_text(json.dumps({**meta, **kb}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
