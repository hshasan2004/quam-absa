"""Fine-tuned Transformer baselines on EComReviews-BD, evaluated exactly like QuAM-xLSTM.

    python scripts/baseline_ft.py --name bl_banglabert_official --model csebuetnlp/banglabert --eco-split official

Same splits, same aspect inventory per category, same losses (presence BCE + polarity CE on
mentioned aspects + 0.5 x overall CE), same metric code (quam/metrics.py) and the same
presence-threshold selection on validation. Training settings follow the EComReviews-BD paper
(AdamW, lr 2e-5, up to 10 epochs, early stopping with patience 3, linear warm-up).

Input: "<category> [SEP] <review>" so the encoder knows the product category, as QuAM does via
its prefix token. One [CLS] representation feeds three heads: presence (one logit per aspect),
polarity (pos/neg per aspect) and overall sentiment (3 classes).

--freeze-embeddings freezes the word-embedding matrix, needed for XLM-R (250k vocabulary) to fit
AdamW states into 4 GB of GPU memory. Writes runs/<name>/{config.json, log.txt, metrics.json,
test_preds.npz} (no checkpoint, to save disk).
"""
import argparse
import json
import math
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
from quam.data import SENT, LengthBucketSampler, domain_query, load_records  # noqa: E402
from quam.metrics import best_eco_threshold, eco_metrics, per_group  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
POS, NEG = SENT["positive"], SENT["negative"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="csebuetnlp/banglabert")
    ap.add_argument("--eco-split", choices=["official", "grouped", "zeroshot"], default="official")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--overall-weight", type=float, default=0.5)
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves GPU memory)")
    ap.add_argument("--freeze-encoder", action="store_true", help="frozen encoder + trained MLP head")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N reviews per part")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


class EcoText(torch.utils.data.Dataset):
    def __init__(self, recs, part, split, tok, aspect_ix, max_len, limit=0):
        self.items = [r for r in recs if r["text"] and r["splits"][split] == part]
        if limit:
            self.items = self.items[:limit]
        self.enc = tok([domain_query(r["domain"]) for r in self.items], [r["text"] for r in self.items],
                       truncation="only_second", max_length=max_len)["input_ids"]
        self.aspect_ix = aspect_ix

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        names = [a["query"] for a in r["aspects"]]
        pres = [int(a["label"] != "none") for a in r["aspects"]]
        sent = [SENT[a["label"]] if a["label"] != "none" else -100 for a in r["aspects"]]
        return {"ids": self.enc[i], "aix": [self.aspect_ix[n] for n in names], "names": names,
                "pres": pres, "sent": sent, "overall": SENT[r["overall"]] if r["overall"] else -100,
                "id": r["id"], "domain": r["domain"]}


def collate(batch, pad_id):
    B, L, A = len(batch), max(len(b["ids"]) for b in batch), max(len(b["aix"]) for b in batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros(B, L, dtype=torch.long)
    aix = torch.zeros(B, A, dtype=torch.long)
    amask = torch.zeros(B, A, dtype=torch.bool)
    pres = torch.full((B, A), -1.0)
    sent = torch.full((B, A), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n, a = len(b["ids"]), len(b["aix"])
        ids[i, :n], att[i, :n] = torch.tensor(b["ids"]), 1
        aix[i, :a], amask[i, :a] = torch.tensor(b["aix"]), True
        pres[i, :a], sent[i, :a] = torch.tensor(b["pres"], dtype=torch.float), torch.tensor(b["sent"])
    return {"ids": ids, "att": att, "aix": aix, "amask": amask, "pres": pres, "sent": sent,
            "overall": torch.tensor([b["overall"] for b in batch]), "meta": batch}


class Baseline(nn.Module):
    def __init__(self, name, n_aspects, freeze_emb, grad_ckpt=False, freeze_enc=False, head_dim=1024):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        if grad_ckpt:          # memory only: recompute activations in backward; results are unchanged
            self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if freeze_emb:
            self.enc.get_input_embeddings().weight.requires_grad_(False)
        self.freeze_enc = freeze_enc
        h = self.enc.config.hidden_size
        if freeze_enc:         # frozen encoder + trained MLP head on LN([CLS; mean]) (same budget class as QuAM)
            self.enc.requires_grad_(False)
            self.mlp = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, head_dim), nn.GELU(), nn.Dropout(0.1))
            h = head_dim
        self.drop = nn.Dropout(0.1)
        self.pres = nn.Linear(h, n_aspects)
        self.pol = nn.Linear(h, n_aspects * 2)       # (pos, neg) per aspect
        self.overall = nn.Linear(h, 3)
        self.n = n_aspects

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc:
            self.enc.eval()
        return self

    def forward(self, ids, att, aix):
        if self.freeze_enc:
            with torch.no_grad():
                hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state.float()
                m = att[..., None].float()
                feat = torch.cat([hs[:, 0], (hs * m).sum(1) / m.sum(1).clamp(min=1)], -1)
            cls = self.drop(self.mlp(feat))
        else:
            cls = self.drop(self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0])
        pres = self.pres(cls).gather(1, aix)                                  # (B, A)
        pol = self.pol(cls).view(-1, self.n, 2)
        pol = pol.gather(1, aix[..., None].expand(-1, -1, 2))                 # (B, A, 2)
        return pres, pol, self.overall(cls)


def to_three(pol):
    """(pos, neg) logits -> 3-class logits in SENT order, neutral impossible."""
    out = pol.new_full(pol.shape[:-1] + (3,), -1e4)
    out[..., POS], out[..., NEG] = pol[..., 0], pol[..., 1]
    return out


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    A = {k: [] for k in ["dataset", "domain", "name", "id", "gold_pres", "prob", "gold_sent", "pred_sent", "verbatim"]}
    R = {k: [] for k in ["dataset", "domain", "id", "gold", "pred"]}
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pres, pol, ov = model(b["ids"].to(dev), b["att"].to(dev), b["aix"].to(dev))
        prob = pres.float().sigmoid().cpu().numpy()
        ps = to_three(pol.float()).argmax(-1).cpu().numpy()
        ovp = ov.float().argmax(-1).cpu().numpy()
        for i, m in enumerate(b["meta"]):
            for j, n in enumerate(m["names"]):
                A["dataset"].append("eco"); A["domain"].append(m["domain"]); A["name"].append(n); A["id"].append(m["id"])
                A["gold_pres"].append(m["pres"][j]); A["prob"].append(prob[i, j])
                A["gold_sent"].append(m["sent"][j]); A["pred_sent"].append(ps[i, j]); A["verbatim"].append(-1)
            R["dataset"].append("eco"); R["domain"].append(m["domain"]); R["id"].append(m["id"])
            R["gold"].append(m["overall"]); R["pred"].append(ovp[i])
    model.train()
    return {k: np.asarray(v) for k, v in A.items()}, {k: np.asarray(v) for k, v in R.items()}


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(dict(vars(args), data="eco", layout="–", readout=args.model,
                                                     presence_mode="fine-tuned", no_domain=False, cons_weight=0),
                                                indent=1), encoding="utf-8")
    logf = open(run / "log.txt", "w", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = "cuda"
    recs = load_records(("eco",))
    names = sorted({a["query"] for r in recs for a in r["aspects"]})
    aspect_ix = {n: i for i, n in enumerate(names)}
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = {p: EcoText(recs, p, args.eco_split, tok, aspect_ix, args.max_len, args.limit) for p in ("train", "val", "test")}
    log(f"{args.model} | split={args.eco_split} train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])} aspects={len(names)}")

    col = lambda b: collate(b, tok.pad_token_id)
    mk = lambda d, bs, sh, seed: torch.utils.data.DataLoader(
        d, batch_sampler=LengthBucketSampler([len(x) for x in d.enc], bs, sh, seed), collate_fn=col)
    train_dl = mk(ds["train"], args.bs, True, args.seed)
    val_dl, test_dl = mk(ds["val"], 64, False, 0), mk(ds["test"], 64, False, 0)

    model = Baseline(args.model, len(names), args.freeze_embeddings, args.grad_ckpt, args.freeze_encoder).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M")
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr)
    total = args.epochs * len(train_dl)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

    best, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(1, args.epochs + 1):
        train_dl.batch_sampler.epoch = epoch
        t0, tot, n = time.time(), 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        for b in train_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pres, pol, ov = model(b["ids"].to(dev), b["att"].to(dev), b["aix"].to(dev))
            pt, st, ot = b["pres"].to(dev), b["sent"].to(dev), b["overall"].to(dev)
            pm, sm, om = pt >= 0, st >= 0, ot >= 0
            loss = F.binary_cross_entropy_with_logits(pres.float()[pm], pt[pm])
            if sm.any():
                loss = loss + F.cross_entropy(to_three(pol.float())[sm], st[sm])
            if om.any():
                loss = loss + args.overall_weight * F.cross_entropy(ov.float()[om], ot[om])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        a, rv = evaluate(model, val_dl, dev)
        t = best_eco_threshold(a, rv)
        m = eco_metrics(a, rv, t)
        log(f"ep {epoch:2d} | loss={tot / n:.3f} | val pairF1={m['pair_F1']:.4f} (P={m['pair_P']:.3f} R={m['pair_R']:.3f} "
            f"t={t:.2f}) detF1={m['det_F1']:.4f} overallF1={m['overall_macroF1']:.4f} | {time.time() - t0:.0f}s "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
        if m["pair_F1"] > best:
            best, best_epoch, bad, best_val = m["pair_F1"], epoch, 0, m
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop (no improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    a, rv = evaluate(model, test_dl, dev)
    thr = best_val["threshold"]
    test = {"eco": eco_metrics(a, rv, thr), "eco_per_domain_pairF1": per_group(a, "domain", "eco", thr)}
    test["select"] = test["eco"]["pair_F1"]
    np.savez_compressed(run / "test_preds.npz", **{f"A_{k}": v for k, v in a.items()}, **{f"R_{k}": v for k, v in rv.items()},
                        threshold=thr)
    (run / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "val": {"eco": best_val}, "test": test},
                                                 indent=1, default=float), encoding="utf-8")
    log("TEST " + json.dumps(test["eco"], default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
