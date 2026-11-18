"""Transformer baseline for BABSA aspect-term sentiment (3 classes) on our grouped split.

Input: the sentence pair (aspect term, text); the [CLS] state goes to a 3-way classifier.
Fine-tuned (default) or frozen encoder + MLP head on LN([CLS; mean]) (--freeze-encoder).
The epoch with the best validation macro-F1 is evaluated on test with quam.metrics.babsa_metrics.

    python scripts/baseline_babsa.py --name bl_babsa_banglabert --model csebuetnlp/banglabert
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import SENT, load_records  # noqa: E402
from quam.metrics import babsa_metrics  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="csebuetnlp/banglabert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def pairs(recs, part, tok, max_len, limit):
    items = [(r, a) for r in recs if r["text"] and r["splits"].get("grouped") == part for a in r["aspects"]]
    if limit:
        items = items[:limit]
    enc = tok([a["name"] for _, a in items], [r["text"] for r, _ in items],
              truncation="only_second", max_length=max_len)["input_ids"]
    meta = [{"id": r["id"], "domain": r["domain"], "label": SENT[a["label"]],
             "verbatim": -1 if a["verbatim"] is None else int(a["verbatim"])} for r, a in items]
    return enc, meta


def batches(enc, meta, bs, shuffle, seed, pad):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    chunks = [order[i: i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.Random(seed).shuffle(chunks)
    for c in chunks:
        L = max(len(enc[i]) for i in c)
        ids = torch.full((len(c), L), pad, dtype=torch.long)
        att = torch.zeros((len(c), L), dtype=torch.long)
        for j, i in enumerate(c):
            ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
        yield ids, att, [meta[i] for i in c]


class Clf(nn.Module):
    def __init__(self, name, freeze_enc, freeze_emb, grad_ckpt):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        if grad_ckpt:
            self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if freeze_emb:
            self.enc.get_input_embeddings().weight.requires_grad_(False)
        h = self.enc.config.hidden_size
        self.freeze_enc = freeze_enc
        if freeze_enc:
            self.enc.requires_grad_(False)
            self.head = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, 1024), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 3))
        else:
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 3))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc:
            self.enc.eval()
        return self

    def forward(self, ids, att):
        if self.freeze_enc:
            with torch.no_grad():
                hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state.float()
                m = att[..., None].float()
                x = torch.cat([hs[:, 0], (hs * m).sum(1) / m.sum(1).clamp(min=1)], -1)
        else:
            x = self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]
        return self.head(x)


@torch.no_grad()
def evaluate(model, data, dev, pad):
    model.eval()
    A = {k: [] for k in ["dataset", "domain", "id", "gold_sent", "pred_sent", "verbatim", "gold_pres", "prob"]}
    for ids, att, meta in batches(*data, 64, False, 0, pad):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = model(ids.to(dev), att.to(dev)).float().argmax(-1).cpu().numpy()
        for m, p in zip(meta, pred):
            A["dataset"].append("babsa"); A["domain"].append(m["domain"]); A["id"].append(m["id"])
            A["gold_sent"].append(m["label"]); A["pred_sent"].append(p); A["verbatim"].append(m["verbatim"])
            A["gold_pres"].append(1); A["prob"].append(1.0)          # presence is not modelled here
    model.train()
    return {k: np.asarray(v) for k, v in A.items()}


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(dict(vars(args), data="babsa"), indent=1), encoding="utf-8")
    logf = open(run / "log.txt", "w", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = args.device
    recs = load_records(("babsa",))
    tok = AutoTokenizer.from_pretrained(args.model)
    data = {p: pairs(recs, p, tok, args.max_len, args.limit) for p in ("train", "val", "test")}
    log(f"{args.model} | BABSA grouped pairs: " + ", ".join(f"{p}={len(v[0])}" for p, v in data.items()))
    model = Clf(args.model, args.freeze_encoder, args.freeze_embeddings, args.grad_ckpt).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M")
    opt = torch.optim.AdamW([{"params": [p for p in trainable if p.dim() > 1], "weight_decay": args.wd},
                             {"params": [p for p in trainable if p.dim() <= 1], "weight_decay": 0.0}], lr=args.lr)
    steps = -(-len(data["train"][0]) // args.bs)
    total, warm = args.epochs * steps, max(1, int(args.warmup * args.epochs * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

    best, best_state, best_epoch, bad, best_val = -1.0, None, -1, 0, None
    for epoch in range(1, args.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        for ids, att, meta in batches(*data["train"], args.bs, True, args.seed + epoch, tok.pad_token_id):
            y = torch.tensor([m["label"] for m in meta], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(ids.to(dev), att.to(dev))
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        m = babsa_metrics(evaluate(model, data["val"], dev, tok.pad_token_id))
        log(f"ep {epoch:2d} | loss={tot / n:.3f} | val acc={m['sent_acc']:.4f} macroF1={m['sent_macroF1']:.4f} | {time.time() - t0:.0f}s")
        if m["sent_macroF1"] > best:
            best, best_epoch, bad, best_val = m["sent_macroF1"], epoch, 0, m
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop (no improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    a = evaluate(model, data["test"], dev, tok.pad_token_id)
    test = babsa_metrics(a)
    test.pop("presence_AUROC", None)
    np.savez_compressed(run / "test_preds.npz", **{f"A_{k}": v for k, v in a.items()})
    (run / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "val": {"babsa": best_val},
                                                  "test": {"babsa": test, "select": test["sent_macroF1"]}},
                                                 indent=1, default=float), encoding="utf-8")
    log("TEST " + json.dumps(test, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
