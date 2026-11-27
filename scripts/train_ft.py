"""QuAM-FT: QuAM-xLSTM on top of a fine-tuned encoder (instead of a frozen, cached one).

    python scripts/train_ft.py --name ft_official --eco-split official
    python scripts/train_ft.py --name ftp_official --eco-split official --prompt-aspects
    python scripts/train_ft.py --name ft_babsa --data babsa --model csebuetnlp/banglabert --prompt-aspects

The encoder reads "<category> </s></s> <review>" and its token states replace the cached ones;
the QuAM head (bidirectional mLSTM memory, name queries, direct per-aspect scorer) is unchanged.

--prompt-aspects (aspect-prompted encoding): the candidate aspect names are written into the
first segment, "<category> : <a1> | <a2> | ... </s></s> <review>", so the encoder reads the
names and the review jointly (still one pass per review). Each aspect's query is the mean of the
encoder states over its own name span, i.e. a contextual name embedding; unseen aspect names are
handled the same way, since they are just text.

Two learning rates: encoder --enc-lr (2e-5, like the fine-tuned baselines), head --lr (1e-3).
Writes runs/<name>/{config.json, log.txt, metrics.json, test_preds.npz} (no checkpoint).
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
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from quam.data import (OVERALL_QUERY, ABSADataset, EncoderCache, LengthBucketSampler, collate,  # noqa: E402
                       domain_query, load_records)
from quam.metrics import per_group  # noqa: E402
from quam.model import QuAM  # noqa: E402
from train import eco_vocab, losses, score, to_dev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PAIRED = Path.home() / ".cache" / "quam-absa" / "xlm-roberta-base-paired"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", choices=["eco", "babsa"], default="eco")
    ap.add_argument("--eco-split", choices=["official", "grouped", "zeroshot"], default="official")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--cache", default=str(PAIRED), help="only used for aspect-name / category query vectors")
    ap.add_argument("--prompt-aspects", action="store_true")
    ap.add_argument("--aspect-kb", default="", help="multi-script aspect knowledge base (JSON); retrieved forms are"
                    " written next to each aspect name in the prompt and read together with it")
    ap.add_argument("--kb-scripts", default="en,bn,bl", help="which KB fields to retrieve: en (English synonyms),"
                    " bn (Bangla script), bl (Banglish)")
    ap.add_argument("--kb-shuffle", action="store_true", help="control: give each aspect the KB forms of another"
                    " (random) aspect -- same extra tokens, wrong meaning")
    ap.add_argument("--kb-k", type=int, default=8, help="max retrieved forms per aspect (scored KB)")
    ap.add_argument("--form-gate", action="store_true",
                    help="RAMP-R evidence-gated read-out: each retrieved form gets its own span vector; a gate "
                         "from (form-review match, retrieval score, form-name agreement) decides how much of it "
                         "is added to the aspect vector, so noisy retrievals can be ignored")
    ap.add_argument("--det-noisy-or", action="store_true",
                    help="detection = noisy-OR with leak: P = 1 - (1 - p0) * prod_k (1 - e_k); p0 = the base detector, "
                         "e_k = evidence that retrieved form k occurs in the review (no match -> P = p0)")
    ap.add_argument("--alias-drop", type=float, nargs=2, default=(0.0, 0.0), metavar=("ALL", "EACH"),
                    help="training: drop all forms of an aspect / each form with these probabilities")
    ap.add_argument("--aspect-read", choices=["span", "marker"], default="span",
                    help="span: mean over the aspect-name tokens; marker: the separator token before the name (TMM-style)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--enc-wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256, help="max review tokens")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze the lowest N encoder layers")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layout", default="mm")
    ap.add_argument("--readout", default="hybrid")
    ap.add_argument("--id-drop", type=float, default=0.2)
    ap.add_argument("--direct", choices=["add", "only", "off"], default="add", help="only: no xLSTM memory (ablation)")
    ap.add_argument("--path-drop", type=float, default=0.0,
                    help="training: hide the direct scores of an aspect with this probability (memory must predict alone)")
    ap.add_argument("--direct-src", choices=["enc", "mem"], default="enc",
                    help="direct scorer features from the encoder states (enc) or from the memory token outputs (mem)")
    ap.add_argument("--overall-weight", type=float, default=0.5)
    ap.add_argument("--babsa-neg", type=int, default=3)
    ap.add_argument("--cons-weight", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048, help="micro-batch token budget (rows x length)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N reviews per part")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


class NoTokens:
    """Cache proxy: query vectors from the real cache, no cached review states (the encoder runs live)."""

    def __init__(self, cache):
        self.c = cache

    def review(self, r):
        return torch.zeros(1, 1)

    def text(self, s):
        return torch.zeros(1, 1)

    def query(self, s):
        return self.c.query(s)


class TextDataset(ABSADataset):
    def __init__(self, *a, tok, max_len, prompt, marker=False, kb=None, kb_score=None, form_gate=False, **kw):
        super().__init__(*a, **kw)
        self.tok, self.max_len, self.prompt, self.marker = tok, max_len, prompt, marker
        self.kb, self.kb_score, self.form_gate = kb or {}, kb_score or {}, form_gate
        self._piece = {}
        self._fkey = {}

    def phi(self, form, grams):
        """String evidence: cross-script phonetic match of a retrieved form anywhere in the review (0..1)."""
        from build_rag_kb import pkey
        from rapidfuzz.distance import Levenshtein
        if form not in self._fkey:
            self._fkey[form] = (form.lower(), pkey(form), len(form.split()))
        f, k, n = self._fkey[form]
        best = 0.0
        for g, gk in grams.get(n, ()):
            if g == f:
                return 1.0
            if len(k) >= 3 and len(gk) >= 3:
                best = max(best, Levenshtein.normalized_similarity(k, gk))
            elif len(k) >= 2 and k == gk:                  # short keys: exact phonetic match only
                best = max(best, 0.9)
        return best

    def entry_pieces(self, name):
        """Token ids of 'name ( f1 ; f2 ; ... )' built piece by piece, with the span of each form."""
        ids, forms = list(self.piece(name)), []
        fs = self.kb.get(name, [])
        if fs:
            ids += self.piece("(")
            for k, f in enumerate(fs):
                if k:
                    ids += self.piece(";")
                p = self.piece(f)
                forms.append((len(ids), len(ids) + len(p)))
                ids += p
            ids += self.piece(")")
        return ids, forms

    def entry(self, name):
        """Aspect name plus its retrieved multi-script forms: 'range ( coverage ; রেঞ্জ ; renj )'."""
        forms = self.kb.get(name)
        return f"{name} ( {' ; '.join(forms)} )" if forms else name

    def piece(self, s):
        if s not in self._piece:
            self._piece[s] = self.tok(s, add_special_tokens=False)["input_ids"]
        return self._piece[s]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        r, t = self.items[idx], self.tok
        ids, spans = [t.cls_token_id] + self.piece(domain_query(r["domain"])), []
        if self.prompt:
            ids += self.piece(":")
            names = item["aspect_names"]
            order = list(range(len(names)))
            if r["dataset"] == "babsa":       # gold aspects come first, sampled absent ones last: hide that order
                random.Random(hash((self.seed, self.epoch if self.train else -1, idx, 7919))).shuffle(order)
            spans = [None] * len(names)
            fspans, fprior, fphi = [[] for _ in names], [[] for _ in names], [[] for _ in names]
            if self.form_gate:
                from build_rag_kb import TOK, pkey
                w = [x.lower() for x in TOK.findall(r["text"])]
                grams = {1: [(x, pkey(x)) for x in set(w)],
                         2: [(g, pkey(g)) for g in {f"{a} {b}" for a, b in zip(w, w[1:])}]}
            for k, j in enumerate(order):
                if k:
                    ids += self.piece("|")
                if self.form_gate:
                    p, fs = self.entry_pieces(names[j])
                    fspans[j] = [(len(ids) + s, len(ids) + e) for s, e in fs]
                    fprior[j] = self.kb_score.get(names[j], [0.0] * len(fs))[: len(fs)]
                    fphi[j] = [self.phi(f, grams) for f in self.kb.get(names[j], [])[: len(fs)]]
                else:
                    p = self.piece(self.entry(names[j]))
                spans[j] = (len(ids) - 1, len(ids)) if self.marker else (len(ids), len(ids) + len(p))
                ids += p
            item["form_spans"], item["form_prior"], item["form_phi"] = fspans, fprior, fphi
        sep2 = [t.sep_token_id, t.sep_token_id] if "roberta" in type(t).__name__.lower() else [t.sep_token_id]
        room = 512 - len(ids) - len(sep2) - 1          # encoder limit (long KB prompts)
        item["rev_start"] = len(ids) + len(sep2)
        ids += sep2 + self.piece(r["text"])[: min(self.max_len, room)] + [t.sep_token_id]
        item["tok_ids"], item["spans"] = ids, spans
        return item


def make_collate(pad_id, prompt):
    def col(batch):
        out = collate(batch)
        B, L, A = len(batch), max(len(b["tok_ids"]) for b in batch), out["aq"].shape[1]
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        att = torch.zeros(B, L, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b["tok_ids"])] = torch.tensor(b["tok_ids"]); att[i, :len(b["tok_ids"])] = 1
        out["tok_ids"], out["att"] = ids, att
        if prompt:
            W = torch.zeros(B, A, L)
            for i, b in enumerate(batch):
                for j, (s, e) in enumerate(b["spans"]):
                    W[i, j, s:e] = 1.0 / max(e - s, 1)
            out["span_w"] = W
            if "form_spans" in batch[0] and any(fs for b in batch for fs in b["form_spans"]):
                K = max(len(fs) for b in batch for fs in b["form_spans"])
                FW, fm, fp = torch.zeros(B, A, K, L), torch.zeros(B, A, K), torch.zeros(B, A, K)
                fphi = torch.zeros(B, A, K)
                rev = torch.zeros(B, L)
                for i, b in enumerate(batch):
                    rev[i, b["rev_start"]:len(b["tok_ids"]) - 1] = 1
                    for j, fs in enumerate(b["form_spans"]):
                        for k, (s, e) in enumerate(fs):
                            FW[i, j, k, s:e] = 1.0 / max(e - s, 1); fm[i, j, k] = 1
                            fp[i, j, k] = b["form_prior"][j][k] if k < len(b["form_prior"][j]) else 0.0
                            fphi[i, j, k] = b["form_phi"][j][k] if k < len(b["form_phi"][j]) else 0.0
                out["form_w"], out["form_mask"], out["form_prior"], out["rev_mask"] = FW, fm, fp, rev
                out["form_phi"] = fphi
        return out
    return col


class QuAMFT(nn.Module):
    def __init__(self, args, cache):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model)
        self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.enc.get_input_embeddings().weight.requires_grad_(False)
        if args.freeze_layers:
            for layer in self.enc.encoder.layer[: args.freeze_layers]:
                layer.requires_grad_(False)
        h = self.enc.config.hidden_size
        stats = dict(cache.stats)
        if args.model != "xlm-roberta-base":       # token statistics of the cache belong to XLM-R
            stats["tok_mean"], stats["tok_std"] = torch.zeros(h), torch.ones(h)
        self.prompt = args.prompt_aspects
        if self.prompt:
            self.ctx_ln = nn.LayerNorm(h)
            self.register_buffer("q_mean", stats["q_mean"].float().clone())
            self.register_buffer("q_sd", stats["q_std"].float().clone())
        self.form_gate = getattr(args, "form_gate", False)
        if self.form_gate:
            # gate_k = sigmoid(w . [evidence_k, prior_k, agree_k] + b); starts trusting forms found in the review
            self.gate = nn.Linear(3, 1)
            with torch.no_grad():
                self.gate.weight.copy_(torch.tensor([[4.0, 0.5, 1.0]])); self.gate.bias.fill_(-2.0)
            self.form_ln = nn.LayerNorm(h)
        self.noisy_or = getattr(args, "det_noisy_or", False)
        self.alias_drop = tuple(getattr(args, "alias_drop", (0.0, 0.0)))
        if self.noisy_or:
            # e_k = sigmoid(w . [centred evidence, phi, prior] + b): starts near 0 unless the form occurs (phi ~ 1)
            self.det_gate = nn.Linear(3, 1)
            with torch.no_grad():
                self.det_gate.weight.copy_(torch.tensor([[1.0, 8.0, 0.2]])); self.det_gate.bias.fill_(-7.0)
        self.head = QuAM(h, cache.query(OVERALL_QUERY), stats, d_model=args.d_model, heads=args.heads,
                         layout=args.layout, readout=args.readout, direct=args.direct,
                         aspect_vocab=eco_vocab() if args.data == "eco" else (), id_drop=args.id_drop,
                         direct_raw=True, path_drop=args.path_drop, direct_src=args.direct_src)

    def forward(self, b):
        hs = self.enc(input_ids=b["tok_ids"], attention_mask=b["att"]).last_hidden_state
        b = dict(b, x=hs, mask=b["att"].bool())
        if self.prompt:
            ctx = torch.einsum("bal,bld->bad", b["span_w"].to(hs.dtype), hs).float()
            if self.form_gate and "form_w" in b:
                if self.training and max(self.alias_drop) > 0:
                    fm = b["form_mask"]
                    keep = (torch.rand(fm.shape[:2], device=fm.device) >= self.alias_drop[0])[..., None]
                    keep = keep & (torch.rand(fm.shape, device=fm.device) >= self.alias_drop[1])
                    b = dict(b, form_mask=fm * keep.to(fm.dtype))
                ctx = ctx + self.gated_forms(hs, ctx, b)
            # the head z-scores queries with the cache statistics; invert that so it sees LN(ctx)
            b["aq"] = self.q_mean + self.q_sd * self.ctx_ln(ctx)
        out = self.head(b)
        if self.noisy_or and getattr(self, "_feats", None) is not None:
            feats, fm = self._feats
            e = torch.sigmoid(self.det_gate(feats).squeeze(-1).float()) * fm          # (B, A, K)
            # log space: log(1-P) = log(1-p0) + sum_k log(1-e_k);  returned as a logit for the usual BCE
            l1m = nn.functional.logsigmoid(-out["presence"].float()) + torch.log1p(-e.clamp(max=1 - 1e-4)).sum(-1)
            l1m = l1m.clamp(max=-1e-6)
            out["presence"] = torch.log(-torch.expm1(l1m)) - l1m
            self._feats = None
        return out

    def gated_forms(self, hs, ctx, b):
        """Evidence-gated sum of the retrieved forms' span vectors (B, A, h)."""
        F_ = torch.einsum("bakl,bld->bakd", b["form_w"].to(hs.dtype), hs).float()          # form vectors
        fm = b["form_mask"]
        Fn = nn.functional.normalize(self.form_ln(F_), dim=-1)
        Hn = nn.functional.normalize(self.form_ln(hs.float()), dim=-1)
        sim = torch.einsum("bakd,bld->bakl", Fn, Hn).masked_fill(b["rev_mask"][:, None, None] == 0, -1.0)
        evidence = sim.amax(-1)                                       # does this form occur in the review?
        agree = (Fn * nn.functional.normalize(self.form_ln(ctx), dim=-1)[:, :, None]).sum(-1)
        feats = torch.stack([evidence, b["form_prior"], agree], -1)
        g = torch.sigmoid(self.gate(feats).squeeze(-1)) * fm
        self.last_gate = g.detach()
        if self.noisy_or:
            # centred evidence: subtract the review's mean state (XLM-R states share one dominant direction,
            # so raw cosines are ~0.98 for every token) + string evidence phi
            rm = b["rev_mask"].to(hs.dtype)[..., None]
            mu = (hs * rm).sum(1, keepdim=True) / rm.sum(1, keepdim=True).clamp(min=1)
            Hc = nn.functional.normalize((hs - mu).float(), dim=-1)
            Fc = nn.functional.normalize(F_ - mu[:, None].float(), dim=-1)
            ev_c = torch.einsum("bakd,bld->bakl", Fc, Hc).masked_fill(b["rev_mask"][:, None, None] == 0, -1.0).amax(-1)
            self._feats = (torch.stack([ev_c, b["form_phi"], b["form_prior"]], -1), fm)
        return (g[..., None] * F_).sum(2) / (1.0 + g.sum(2, keepdim=True))


def slice_batch(b, s, e):
    """Rows s:e of a collated batch; trims padding to the longest review/aspect list in the slice."""
    if s == 0 and e == len(b["ids"]):
        return b
    out = {}
    for k, v in b.items():
        if k == "en_rows" or k.endswith("_en"):
            continue
        out[k] = v[s:e] if (torch.is_tensor(v) or isinstance(v, list)) else v
    L = int(out["att"].sum(1).max())
    out["tok_ids"], out["att"] = out["tok_ids"][:, :L], out["att"][:, :L]
    if "span_w" in out:
        out["span_w"] = out["span_w"][:, :, :L]
    if "form_w" in out:
        out["form_w"], out["rev_mask"] = out["form_w"][..., :L], out["rev_mask"][:, :L]
    return out


@torch.no_grad()
def evaluate(model, loader, dev):
    from quam.metrics import Collector
    model.eval()
    col = Collector()
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(to_dev(batch, dev))
        col.add(batch, out)
    model.train()
    return col.arrays()


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    logf = open(run / "log.txt", "w", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = "cuda"
    cache = EncoderCache(args.cache)
    recs = load_records((args.data,))
    pool = sorted({a["name"] for r in recs if r["dataset"] == "babsa" and r["splits"].get("grouped") == "train"
                   for a in r["aspects"]})
    for r in recs:
        r["splits"]["_run"] = r["splits"][args.eco_split] if r["dataset"] == "eco" else r["splits"]["grouped"]
    tok = AutoTokenizer.from_pretrained(args.model)
    kb = kb_score = None
    if args.aspect_kb:
        fields = args.kb_scripts.split(",")
        raw = json.loads(Path(args.aspect_kb).read_text(encoding="utf-8"))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        if raw and isinstance(next(iter(raw.values())), list):       # retrieved (scored) KB from build_rag_kb.py
            kb = {k: [x["form"] for x in v[: args.kb_k]] for k, v in raw.items()}
            kb_score = {k: [float(x["score"]) for x in v[: args.kb_k]] for k, v in raw.items()}
        else:
            kb = {k: [x for fl in fields for x in v.get(fl, [])] for k, v in raw.items()}
            kb_score = {k: [1.0] * len(v) for k, v in kb.items()}
        if args.kb_shuffle:
            keys = sorted(kb)
            perm = keys[:]
            rng = random.Random(0)
            while any(a == b for a, b in zip(keys, perm)):      # derangement: no aspect keeps its own forms
                rng.shuffle(perm)
            kb = {k: kb[p] for k, p in zip(keys, perm)}
        log(f"aspect KB: {len(kb)} entries, fields {fields}, shuffled={args.kb_shuffle}")
    kw = dict(kb=kb, kb_score=kb_score if kb else None, form_gate=args.form_gate, babsa_pool=pool, babsa_neg=args.babsa_neg, seed=args.seed, tok=tok, max_len=args.max_len,
              prompt=args.prompt_aspects, marker=args.aspect_read == "marker", limit=args.limit or None)
    proxy = NoTokens(cache)
    ds = {p: TextDataset(recs, proxy, "_run", p, train=(p == "train"), **kw) for p in ("train", "val", "test")}
    log(f"{args.model} prompt={args.prompt_aspects} | train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}")

    col = make_collate(tok.pad_token_id, args.prompt_aspects)
    lens = lambda d: [len(it["text"]) for it in d.items]
    mk = lambda d, bs, sh, seed: torch.utils.data.DataLoader(
        d, batch_sampler=LengthBucketSampler(lens(d), bs, sh, seed), collate_fn=col, num_workers=0)
    train_dl, val_dl, test_dl = mk(ds["train"], args.bs, True, args.seed), mk(ds["val"], 32, False, 0), mk(ds["test"], 32, False, 0)

    model = QuAMFT(args, cache).to(dev)
    enc_p = [p for p in model.enc.parameters() if p.requires_grad]
    head_p = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    groups = [
        {"params": [p for p in enc_p if p.dim() > 1], "lr": args.enc_lr, "weight_decay": args.enc_wd},
        {"params": [p for p in enc_p if p.dim() <= 1], "lr": args.enc_lr, "weight_decay": 0.0},
        {"params": [p for n, p in head_p if p.dim() > 1], "lr": args.lr, "weight_decay": args.wd},
        {"params": [p for n, p in head_p if p.dim() <= 1], "lr": args.lr, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups)
    trainable = enc_p + [p for _, p in head_p]
    log(f"trainable: encoder {sum(p.numel() for p in enc_p) / 1e6:.1f}M, head {sum(p.numel() for _, p in head_p) / 1e6:.1f}M")
    total = args.epochs * len(train_dl)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

    best, best_state, best_epoch, bad, best_val = -1.0, None, -1, 0, None
    for epoch in range(1, args.epochs + 1):
        ds["train"].epoch = train_dl.batch_sampler.epoch = epoch
        t0, agg, n, skipped = time.time(), {}, 0, 0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            B, L = batch["tok_ids"].shape
            step = max(1, args.max_tokens // L)
            ok = True
            for s in range(0, B, step):             # micro-batches for long reviews (4 GB GPU)
                b = to_dev(slice_batch(batch, s, min(B, s + step)), dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(b)
                    loss, parts = losses(model, out, b, args)
                if not torch.isfinite(loss):
                    ok = False; break
                (loss * (b["tok_ids"].shape[0] / B)).backward()
            if not ok:
                skipped += 1; sched.step(); continue
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(gn):
                skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        a, rv = evaluate(model, val_dl, dev)
        m = score(a, rv, args.data)
        parts = " ".join(f"{k}={v / max(n, 1):.3f}" for k, v in agg.items()) + (f" SKIPPED={skipped}" if skipped else "")
        if "eco" in m:
            e = m["eco"]
            brief = (f"pairF1={e['pair_F1']:.4f} (P={e['pair_P']:.3f} R={e['pair_R']:.3f} t={e['threshold']:.2f}) "
                     f"detF1={e['det_F1']:.4f} detmF1={e['det_macroF1']:.4f} overallF1={e['overall_macroF1']:.4f}")
        else:
            bm = m["babsa"]
            brief = f"babsa sentF1={bm['sent_macroF1']:.4f} acc={bm['sent_acc']:.4f}"
        log(f"ep {epoch:2d} | {parts} | {brief} | {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
        if m["select"] > best:
            best, best_epoch, bad, best_val = m["select"], epoch, 0, m
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop (no improvement for {args.patience} epochs)"); break

    torch.save(best_state, run / "best.pt")             # so a crash in the test pass does not lose training
    model.load_state_dict(best_state)
    for attempt in range(3):                             # 4 GB GPU: transient CUDA errors happen; retry smaller
        try:
            a, rv = evaluate(model, test_dl if attempt == 0 else mk(ds["test"], 8, False, 0), dev)
            break
        except RuntimeError as err:
            log(f"test pass failed ({err}); retry {attempt + 1}")
            torch.cuda.empty_cache(); time.sleep(30)
    else:
        raise RuntimeError("test pass failed 3 times; best.pt is saved")
    thr = best_val.get("eco", {}).get("threshold")
    test = score(a, rv, args.data, threshold=thr)
    if "eco" in test:
        test["eco_per_domain_pairF1"] = per_group(a, "domain", "eco", thr)
    np.savez_compressed(run / "test_preds.npz", **{f"A_{k}": v for k, v in a.items()},
                        **{f"R_{k}": v for k, v in rv.items()}, threshold=thr if thr is not None else -1)
    (run / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "val": best_val, "test": test},
                                                 indent=1, default=float), encoding="utf-8")
    log("TEST " + json.dumps({k: v for k, v in test.items() if "per_domain" not in k}, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
