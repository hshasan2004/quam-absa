"""Run a frozen encoder once over every text, aspect query and domain name; cache the outputs.

Texts keep token-level hidden states (needed by the xLSTM). Queries and domain names are
mean-pooled to one vector each. Token states go to a memory-mapped float16 file so training
never has to hold them all in RAM (this laptop has ~7 GB).

Output directory (default %USERPROFILE%/.cache/quam-absa/<model>):
  texts.npy        (n_tokens, dim) float16, opened with mmap_mode="r"
  texts_index.json {"strings": [...], "offsets": [...], "lengths": [...], "dim": D}
  queries.pt       {"strings": [...], "emb": FloatTensor (n, dim)}
  meta.json        encoder name, max_len, truncation stats
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import OVERALL_QUERY, PAIR_SEP, domain_query, load_records  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def collect(records, pair=False):
    texts, queries = {}, {OVERALL_QUERY: None}
    for r in records:
        if pair:      # key "<category><SEP><review>": the review is encoded together with its category
            if r["text"]:
                texts.setdefault(domain_query(r["domain"]) + PAIR_SEP + r["text"], None)
        else:
            texts.setdefault(r["text"], None)
            if r.get("text_en"):
                texts.setdefault(r["text_en"], None)
        queries.setdefault(domain_query(r["domain"]), None)
        for a in r["aspects"]:
            queries.setdefault(a["query"], None)
    return [t for t in texts if t], list(queries)


@torch.no_grad()
def encode_texts(model, tok, texts, out_dir, max_len, batch_tokens, device):
    if PAIR_SEP in texts[0]:      # paired cache: "<s> category </s></s> review </s>", review truncated
        parts = [t.split(PAIR_SEP, 1) for t in texts]
        full = np.array([len(tok(c, t)["input_ids"]) for c, t in parts])
        ids = [tok(c, t, truncation="only_second", max_length=max_len)["input_ids"] for c, t in parts]
    else:
        ids = [tok(t, truncation=False, add_special_tokens=True)["input_ids"] for t in texts]
        full = np.array([len(x) for x in ids])
        ids = [x if len(x) <= max_len else x[: max_len - 1] + [tok.sep_token_id] for x in ids]
    lengths = np.array([len(x) for x in ids])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    dim = model.config.hidden_size
    mm = np.lib.format.open_memmap(out_dir / "texts.npy", mode="w+", dtype=np.float16,
                                   shape=(int(lengths.sum()), dim))
    order = np.argsort(-lengths)
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        L = lengths[order[i]]
        n = max(1, batch_tokens // L)
        batch = order[i: i + n]
        i += n
        pad = tok.pad_token_id
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            x[j, : lengths[b]] = torch.tensor(ids[b])
            att[j, : lengths[b]] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(input_ids=x.to(device), attention_mask=att.to(device)).last_hidden_state
        h = h.float().cpu().numpy().astype(np.float16)
        for j, b in enumerate(batch):
            mm[offsets[b]: offsets[b] + lengths[b]] = h[j, : lengths[b]]
        done += len(batch)
        if done % 5000 < len(batch):
            print(f"  texts {done}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    del mm
    index = {"strings": texts, "offsets": offsets.tolist(), "lengths": lengths.tolist(), "dim": dim}
    (out_dir / "texts_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return full


@torch.no_grad()
def encode_queries(model, tok, queries, device, bs=256):
    out = []
    for i in range(0, len(queries), bs):
        enc = tok(queries[i: i + bs], padding=True, truncation=True, max_length=32,
                  return_tensors="pt", return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state.float()
        keep = (enc["attention_mask"].bool() & ~special.to(device).bool()).float()[..., None]
        out.append(((h * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pair", action="store_true", help="encode each review together with its category name")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (Path.home() / ".cache" / "quam-absa" /
                                           (args.model.split("/")[-1] + ("-paired" if args.pair else "")))
    out.mkdir(parents=True, exist_ok=True)

    records = load_records()
    texts, queries = collect(records, args.pair)
    print(f"{len(texts)} unique texts, {len(queries)} unique queries -> {out}")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    full = encode_texts(model, tok, texts, out, args.max_len, args.batch_tokens, device)
    q = encode_queries(model, tok, queries, device)
    torch.save({"strings": queries, "emb": q}, out / "queries.pt")
    meta = {"model": args.model, "paired": args.pair, "max_len": args.max_len, "n_texts": len(texts),
            "n_tokens": int(np.minimum(full, args.max_len).sum()),
            "truncated_texts": int((full > args.max_len).sum()),
            "token_len_p50_p95_p99_max": [int(np.percentile(full, p)) for p in (50, 95, 99)] + [int(full.max())]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
