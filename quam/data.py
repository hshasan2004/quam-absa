"""Loading processed records, the encoder cache, and batching for QuAM training."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from .textutils import aspect_query

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OVERALL_QUERY = "overall opinion"
SCRIPTS = ["bangla", "english", "banglish", "mixed", "other"]
SENT = {"positive": 0, "neutral": 1, "negative": 2}
SENT_NAMES = ["positive", "neutral", "negative"]


def domain_query(domain: str) -> str:
    """'Mini_Printer' -> 'mini printer', 'Politics / Social Issue' -> 'politics / social issue'."""
    return aspect_query(domain)


def load_records(which=("eco", "babsa")):
    files = {"eco": "ecomreviews.jsonl", "babsa": "babsa.jsonl"}
    out = []
    for w in which:
        with open(PROCESSED / files[w], encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f)
    return out


def default_cache_dir(model="xlm-roberta-base"):
    return Path.home() / ".cache" / "quam-absa" / model


class EncoderCache:
    """Read-only view of scripts/encode.py output; token states stay on disk (memmap)."""

    def __init__(self, path):
        path = Path(path)
        idx = json.loads((path / "texts_index.json").read_text(encoding="utf-8"))
        self.text_id = {s: i for i, s in enumerate(idx["strings"])}
        self.offsets, self.lengths, self.dim = idx["offsets"], idx["lengths"], idx["dim"]
        self.tokens = np.load(path / "texts.npy", mmap_mode="r")
        q = torch.load(path / "queries.pt")
        self.query_id = {s: i for i, s in enumerate(q["strings"])}
        self.query_emb = q["emb"].float()
        self.stats = self._feature_stats(path)

    def _feature_stats(self, path, n_sample=200_000, seed=0):
        """Per-feature mean/std of token states and of query vectors.

        XLM-R has 'rogue' dimensions (one feature mean ~18 vs typical std ~0.08) that make all
        vectors nearly parallel (aspect-query cosine 0.989). Standardising per feature fixes it.
        """
        f = path / "feature_stats.pt"
        if f.exists():
            return torch.load(f)
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(self.tokens.shape[0], size=min(n_sample, self.tokens.shape[0]), replace=False))
        tok = torch.from_numpy(np.asarray(self.tokens[rows], dtype=np.float32))
        stats = {"tok_mean": tok.mean(0), "tok_std": tok.std(0).clamp(min=1e-4),
                 "q_mean": self.query_emb.mean(0), "q_std": self.query_emb.std(0).clamp(min=1e-4)}
        torch.save(stats, f)
        return stats

    def text(self, s) -> torch.Tensor:
        i = self.text_id[s]
        o, n = self.offsets[i], self.lengths[i]
        return torch.from_numpy(np.array(self.tokens[o: o + n]))

    def query(self, s) -> torch.Tensor:
        return self.query_emb[self.query_id[s]]


class ABSADataset(torch.utils.data.Dataset):
    """One item per review. Aspect targets:

    presence: 1 mentioned, 0 not mentioned, -1 ignore
    sentiment: index into SENT_NAMES, -100 ignore (absent aspects)
    EComReviews never labels an aspect neutral, so its sentiment is restricted to pos/neg.
    For BABSA (only mentioned aspects are annotated) we add `babsa_neg` sampled absent
    aspects per review: terms from other reviews that do not occur in this text.
    """

    def __init__(self, records, cache, split, part, babsa_pool=None, babsa_neg=3,
                 train=False, seed=0, use_en=False, limit=None):
        self.items = [r for r in records if r["text"] and r["splits"].get(split) == part]
        if limit:
            self.items = self.items[:limit]
        self.cache, self.train, self.seed, self.use_en = cache, train, seed, use_en
        self.pool, self.babsa_neg = babsa_pool or [], babsa_neg
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def _negatives(self, r, idx):
        if not self.pool or self.babsa_neg <= 0:
            return []
        rng = random.Random(hash((self.seed, self.epoch if self.train else -1, idx)))
        gold = {a["name"] for a in r["aspects"]}
        out = []
        for _ in range(self.babsa_neg * 10):
            a = rng.choice(self.pool)
            if a not in gold and a not in r["text"] and a not in out:
                out.append(a)
                if len(out) == self.babsa_neg:
                    break
        return out

    def __getitem__(self, idx):
        r = self.items[idx]
        names, pres, sent, verb = [], [], [], []
        for a in r["aspects"]:
            names.append(a["query"])
            if r["dataset"] == "eco":
                present = a["label"] != "none"
                pres.append(int(present))
                sent.append(SENT[a["label"]] if present else -100)
            else:
                pres.append(1)
                sent.append(SENT[a["label"]])
            verb.append(-1 if a["verbatim"] is None else int(a["verbatim"]))
        if r["dataset"] == "babsa":
            for a in self._negatives(r, idx):
                names.append(a); pres.append(0); sent.append(-100); verb.append(-1)
        en = r.get("text_en") if self.use_en else None
        return {
            "id": r["id"], "dataset": r["dataset"], "domain": r["domain"],
            "x": self.cache.text(r["text"]),
            "x_en": self.cache.text(en) if en else None,
            "dom_q": self.cache.query(domain_query(r["domain"])),
            "script": SCRIPTS.index(r["script"]),
            "aspect_names": names,
            "aq": torch.stack([self.cache.query(n) for n in names]),
            "presence": torch.tensor(pres, dtype=torch.float),
            "sentiment": torch.tensor(sent, dtype=torch.long),
            "verbatim": torch.tensor(verb, dtype=torch.long),
            "allow_neutral": r["dataset"] == "babsa",
            "overall": SENT[r["overall"]] if r["overall"] else -100,
        }


def _pad_tokens(seqs):
    L = max(s.shape[0] for s in seqs)
    x = torch.zeros(len(seqs), L, seqs[0].shape[1], dtype=torch.float16)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return x, mask


def collate(batch):
    B, A = len(batch), max(len(b["aspect_names"]) for b in batch)
    D = batch[0]["aq"].shape[1]
    x, mask = _pad_tokens([b["x"] for b in batch])
    out = {
        "x": x, "mask": mask,
        "dom_q": torch.stack([b["dom_q"] for b in batch]),
        "script": torch.tensor([b["script"] for b in batch]),
        "aq": torch.zeros(B, A, D), "a_mask": torch.zeros(B, A, dtype=torch.bool),
        "presence": torch.full((B, A), -1.0), "sentiment": torch.full((B, A), -100, dtype=torch.long),
        "verbatim": torch.full((B, A), -1, dtype=torch.long),
        "allow_neutral": torch.tensor([b["allow_neutral"] for b in batch]),
        "overall": torch.tensor([b["overall"] for b in batch]),
        "ids": [b["id"] for b in batch], "datasets": [b["dataset"] for b in batch],
        "domains": [b["domain"] for b in batch], "aspect_names": [b["aspect_names"] for b in batch],
    }
    for i, b in enumerate(batch):
        n = len(b["aspect_names"])
        out["aq"][i, :n] = b["aq"]
        out["a_mask"][i, :n] = True
        out["presence"][i, :n] = b["presence"]
        out["sentiment"][i, :n] = b["sentiment"]
        out["verbatim"][i, :n] = b["verbatim"]
    en_rows = [i for i, b in enumerate(batch) if b["x_en"] is not None]
    if en_rows:
        out["x_en"], out["mask_en"] = _pad_tokens([batch[i]["x_en"] for i in en_rows])
        out["en_rows"] = torch.tensor(en_rows)
    return out


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar length (less padding); shuffled across batches each epoch."""

    def __init__(self, lengths, batch_size, shuffle, seed=0):
        self.lengths, self.bs, self.shuffle, self.seed, self.epoch = np.asarray(lengths), batch_size, shuffle, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        chunks = [idx[i: i + self.bs * 50] for i in range(0, len(idx), self.bs * 50)]
        batches = []
        for c in chunks:
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[i: i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
