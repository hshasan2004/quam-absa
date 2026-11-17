"""Train and evaluate QuAM-xLSTM.

Examples
  python scripts/train.py --data eco --eco-split official --name eco_official
  python scripts/train.py --data babsa --name babsa
  python scripts/train.py --data eco --overfit 64 --epochs 60      # sanity check
Outputs go to runs/<name>/: config.json, best.pt, metrics.json, log.txt
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import (OVERALL_QUERY, SCRIPTS, ABSADataset, EncoderCache, LengthBucketSampler,  # noqa: E402
                       collate, default_cache_dir, load_records)
from quam.metrics import Collector, babsa_metrics, best_eco_threshold, eco_metrics, per_group  # noqa: E402
from quam.model import PRESENCE_MODES, QUERY_INITS, QuAM  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", choices=["eco", "babsa", "both"], default="eco")
    ap.add_argument("--eco-split", choices=["official", "grouped", "zeroshot"], default="official")
    ap.add_argument("--cache", default=str(default_cache_dir()))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layout", default="mmsm")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--presence-mode", choices=PRESENCE_MODES, default="mag+value")
    ap.add_argument("--readout", choices=["memory", "meanpool"], default="memory")
    ap.add_argument("--no-domain", action="store_true")
    ap.add_argument("--query-init", choices=QUERY_INITS, default="free",
                    help="tied: aspect query starts as the memory's own key for the aspect name")
    ap.add_argument("--overall-weight", type=float, default=0.5)
    ap.add_argument("--cons-weight", type=float, default=0.0, help="cross-lingual consistency (eco English pairs)")
    ap.add_argument("--babsa-neg", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--overfit", type=int, default=0, help="train+eval on the first N training reviews")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def losses(model, out, batch, args):
    pres_t, sent_t, ov_t = batch["presence"], batch["sentiment"], batch["overall"]
    pm = pres_t >= 0
    l_pres = F.binary_cross_entropy_with_logits(out["presence"][pm].float(), pres_t[pm])
    sm = sent_t >= 0
    l_sent = F.cross_entropy(out["sentiment"][sm].float(), sent_t[sm]) if sm.any() else out["presence"].sum() * 0
    om = ov_t >= 0
    l_ov = F.cross_entropy(out["overall"][om].float(), ov_t[om]) if om.any() else out["presence"].sum() * 0
    total = l_pres + l_sent + args.overall_weight * l_ov
    parts = {"pres": l_pres.item(), "sent": l_sent.item(), "overall": l_ov.item()}

    if args.cons_weight > 0 and "x_en" in batch:
        rows = batch["en_rows"]
        en = {"x": batch["x_en"], "mask": batch["mask_en"], "dom_q": batch["dom_q"][rows],
              "script": torch.full_like(batch["script"][rows], SCRIPTS.index("english")),
              "aq": batch["aq"][rows], "a_mask": batch["a_mask"][rows], "allow_neutral": batch["allow_neutral"][rows]}
        oe = model(en)
        am = batch["a_mask"][rows]
        po, pe = out["presence"][rows][am].float(), oe["presence"][am].float()
        bern = lambda a, b: (torch.sigmoid(a) * (F.logsigmoid(a) - F.logsigmoid(b))
                             + torch.sigmoid(-a) * (F.logsigmoid(-a) - F.logsigmoid(-b)))
        l_c = 0.5 * (bern(po, pe) + bern(pe, po)).mean()
        gm = sent_t[rows] >= 0
        if gm.any():
            so, se = out["sentiment"][rows][gm].float().log_softmax(-1), oe["sentiment"][gm].float().log_softmax(-1)
            l_c = l_c + 0.5 * (F.kl_div(se, so, log_target=True, reduction="batchmean")
                               + F.kl_div(so, se, log_target=True, reduction="batchmean"))
        total = total + args.cons_weight * l_c
        parts["cons"] = l_c.item()
    return total, parts


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    col = Collector()
    for batch in loader:
        b = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        col.add(batch, out)
    model.train()
    return col.arrays()


def score(a, rv, data, threshold=None):
    m = {}
    if data in ("eco", "both"):
        t = best_eco_threshold(a, rv) if threshold is None else threshold
        m["eco"] = eco_metrics(a, rv, t)
    if data in ("babsa", "both"):
        m["babsa"] = babsa_metrics(a)
    sel = [m["eco"]["pair_F1"]] if "eco" in m else []
    sel += [m["babsa"]["sent_macroF1"]] if "babsa" in m else []
    m["select"] = float(np.mean(sel))
    return m


def make_loader(ds, bs, shuffle, seed):
    lengths = [it["text"].__len__() for it in ds.items]
    return torch.utils.data.DataLoader(ds, batch_sampler=LengthBucketSampler(lengths, bs, shuffle, seed),
                                       collate_fn=collate, num_workers=0, pin_memory=True)


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    logf = open(run / "log.txt", "w", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    dev = "cuda"
    cache = EncoderCache(args.cache)
    which = {"eco": ("eco",), "babsa": ("babsa",), "both": ("eco", "babsa")}[args.data]
    recs = load_records(which)
    pool = sorted({a["name"] for r in recs if r["dataset"] == "babsa" and r["splits"].get("grouped") == "train"
                   for a in r["aspects"]})

    def split_of(r):
        return r["splits"][args.eco_split] if r["dataset"] == "eco" else r["splits"]["grouped"]

    for r in recs:
        r["splits"]["_run"] = split_of(r)
    kw = dict(babsa_pool=pool, babsa_neg=args.babsa_neg, seed=args.seed)
    train_ds = ABSADataset(recs, cache, "_run", "train", train=True, use_en=args.cons_weight > 0,
                           limit=args.overfit or None, **kw)
    if args.overfit:
        val_ds = test_ds = ABSADataset(recs, cache, "_run", "train", limit=args.overfit, **kw)
    else:
        val_ds = ABSADataset(recs, cache, "_run", "val", **kw)
        test_ds = ABSADataset(recs, cache, "_run", "test", **kw)
    log(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  babsa negative pool={len(pool)}")

    train_dl = make_loader(train_ds, args.bs, True, args.seed)
    val_dl, test_dl = make_loader(val_ds, 64, False, 0), make_loader(test_ds, 64, False, 0)

    model = QuAM(cache.dim, cache.query(OVERALL_QUERY), cache.stats, d_model=args.d_model, heads=args.heads,
                 layout=args.layout, dropout=args.dropout, presence_mode=args.presence_mode,
                 use_domain=not args.no_domain, readout=args.readout, query_init=args.query_init).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"QuAM params: {n_params / 1e6:.2f}M  layout={args.layout} d={args.d_model} heads={args.heads}")

    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total_steps = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    best, best_epoch, bad = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        train_ds.epoch = train_dl.batch_sampler.epoch = epoch
        t0, agg, n, skipped = time.time(), {}, 0, 0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_dl:
            b = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b)
                loss, parts = losses(model, out, b, args)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                sched.step()
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        a, rv = evaluate(model, val_dl, dev)
        m = score(a, rv, args.data)
        parts = " ".join(f"{k}={v / max(n, 1):.3f}" for k, v in agg.items())
        if skipped:
            parts += f" SKIPPED_NONFINITE={skipped}"
        brief = []
        if "eco" in m:
            e = m["eco"]
            brief.append(f"eco pairF1={e['pair_F1']:.4f} (P={e['pair_P']:.3f} R={e['pair_R']:.3f} t={e['threshold']:.2f}) "
                         f"detF1={e['det_F1']:.4f} overallF1={e['overall_macroF1']:.4f}")
        if "babsa" in m:
            bm = m["babsa"]
            brief.append(f"babsa sentF1={bm['sent_macroF1']:.4f} acc={bm['sent_acc']:.4f} presAUC={bm.get('presence_AUROC', float('nan')):.4f}")
        log(f"ep {epoch:2d} | {parts} | {' | '.join(brief)} | {time.time() - t0:.0f}s "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
        if m["select"] > best:
            best, best_epoch, bad = m["select"], epoch, 0
            torch.save(model.state_dict(), run / "best.pt")
            best_val = m
        else:
            bad += 1
            if bad >= args.patience and not args.overfit:
                log(f"early stop (no improvement for {args.patience} epochs)")
                break

    model.load_state_dict(torch.load(run / "best.pt"))
    a, rv = evaluate(model, test_dl, dev)
    thr = best_val.get("eco", {}).get("threshold")
    test = score(a, rv, args.data, threshold=thr)
    if "eco" in test:
        test["eco_per_domain_pairF1"] = per_group(a, "domain", "eco", thr)
    if "babsa" in test:
        test["babsa_per_domain_sentF1"] = per_group(a, "domain", "babsa")
    result = {"best_epoch": best_epoch, "val": best_val, "test": test, "params": n_params}
    (run / "metrics.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log("TEST " + json.dumps({k: v for k, v in test.items() if not k.endswith("per_domain_pairF1")
                              and not k.endswith("per_domain_sentF1")}, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
