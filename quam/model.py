"""QuAM-xLSTM: Query-the-Aspect-Memory.

  review tokens (frozen encoder) ─► [domain+script prefix] ─► Bi-xLSTM stack ─► final mLSTM memory
  aspect name (frozen encoder)   ─► query projection ───────────────────────────► read memory
      read gives  r       = C q / max(|n^T q|, e^-m)   -> sentiment head (pos / neu / neg)
                  log_mag = log |n^T q|                -> presence head (mentioned or not)
  overall sentiment = aggregation of aspect sentiments, weighted by presence and a learned
                      per-(domain, aspect) importance, plus a read with the query "overall opinion".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SCRIPTS
from .xlstm import BiXLSTMStack, mlstm_read

PRESENCE_MODES = ("mag+value", "mag", "value")
QUERY_INITS = ("free", "tied")
# memory:   everything from the memory read
# meanpool: everything from mean-pooled tokens + aspect query (AF-MAT style ablation)
# hybrid:   presence from memory read + pooled tokens; sentiment and overall from the memory read
READOUTS = ("memory", "meanpool", "hybrid")
# direct path: g = MLP(LN([x_<s>; mean x])) of the frozen encoder states, scored with per-aspect weights
#   score_a = E[id(a)].g + MLP([g; q_a])  (learned row for known aspects + a name-conditioned part; unseen
#   aspects have id 0 and use the name part only).  off: not used; add: added to the memory heads' logits;
#   only: replaces them (ablation: is the memory read useful on top of the direct path?)
DIRECT_MODES = ("off", "add", "only")


class AspectScorer(nn.Module):
    """Per-aspect scorer over the direct features g, k outputs per aspect:
    logits = E[id(a)] . g  (a learned row per known aspect, like a linear output layer)
           + MLP([g; q_a])  (from the aspect name, so unseen aspects (id 0, row = 0) still get scores)."""

    def __init__(self, n_ids, q_dim, g_dim, k, hidden=256):
        super().__init__()
        self.k, self.g_dim = k, g_dim
        self.emb = nn.Embedding(n_ids + 1, g_dim * k, padding_idx=0)
        nn.init.zeros_(self.emb.weight)                     # unseen ids stay exactly zero
        self.name = nn.Sequential(nn.Linear(g_dim + q_dim, hidden), nn.GELU(), nn.Linear(hidden, k))

    def forward(self, g, q, ids):                           # g (B, G), q (B, A, Q), ids (B, A)
        w = self.emb(ids).view(*ids.shape, self.k, self.g_dim)
        by_id = torch.einsum("bg,bakg->bak", g, w)
        by_name = self.name(torch.cat([g[:, None].expand(-1, q.shape[1], -1), q], -1))
        return by_id + by_name


def mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class Standardize(nn.Module):
    """Fixed per-feature z-scoring with statistics from the encoder cache."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("std", std.float().clone())

    def forward(self, x):
        return (x.float() - self.mean) / self.std


class QuAM(nn.Module):
    def __init__(self, enc_dim, overall_query, stats, d_model=256, heads=4, layout="mmsm", dropout=0.1,
                 presence_mode="mag+value", use_domain=True, readout="memory", query_init="free",
                 direct="off", aspect_vocab=(), direct_dim=1024, id_drop=0.2, direct_raw=False):
        super().__init__()
        assert presence_mode in PRESENCE_MODES and readout in READOUTS
        assert query_init in QUERY_INITS and direct in DIRECT_MODES
        self.direct, self.id_drop, self.direct_raw = direct, id_drop, direct_raw
        if direct != "off":
            self.aspect_vocab = {n: i + 1 for i, n in enumerate(aspect_vocab)}
            self.g = nn.Sequential(nn.LayerNorm(2 * enc_dim), nn.Linear(2 * enc_dim, direct_dim), nn.GELU(),
                                   nn.Dropout(dropout))
            self.d_pres = AspectScorer(len(aspect_vocab), enc_dim, direct_dim, 1)
            self.d_sent = AspectScorer(len(aspect_vocab), enc_dim, direct_dim, 3)
            self.d_overall = nn.Linear(direct_dim, 3)
        self.presence_mode, self.use_domain, self.readout, self.heads = presence_mode, use_domain, readout, heads
        self.tied = query_init == "tied"
        inner = 2 * d_model                    # mLSTM up-projection factor 2
        self.inner = inner
        self.tok_std = Standardize(stats["tok_mean"], stats["tok_std"])
        self.q_std = Standardize(stats["q_mean"], stats["q_std"])
        self.inp = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model), nn.Dropout(dropout))
        self.dom = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model))
        self.script = nn.Embedding(len(SCRIPTS), d_model)
        self.stack = BiXLSTMStack(d_model, heads, layout, dropout)
        self.query = nn.Sequential(nn.LayerNorm(enc_dim), mlp(enc_dim, d_model, 2 * inner))  # fwd + bwd
        if self.tied:
            # tied: query = key_of(aspect) + learned correction, which starts at exactly zero
            nn.init.zeros_(self.query[1][-1].weight)
            nn.init.zeros_(self.query[1][-1].bias)
        self.asp_proj = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model))
        self.register_buffer("overall_q", overall_query.float().clone(), persistent=False)

        r_dim = 2 * inner
        self.r_norm = nn.LayerNorm(r_dim)
        if readout in ("meanpool", "hybrid"):  # pooled tokens + aspect query
            self.pool_proj = nn.Linear(d_model + 2 * inner, r_dim)
        if readout == "hybrid":
            self.p_norm = nn.LayerNorm(r_dim)
        self.sent_head = mlp(r_dim, d_model, 3, dropout)
        mag_dim = 2 * heads + 1                # log|n^T q| per head and direction, plus log length
        pres_in = {"mag+value": mag_dim + r_dim, "mag": mag_dim, "value": r_dim}[presence_mode]
        pres_in += r_dim if readout == "hybrid" else 0
        self.pres_head = mlp(pres_in, d_model // 2, 1, dropout)
        self.importance = mlp(2 * d_model, d_model // 2, 1)
        self.agg = mlp(4, 32, 3)
        self.overall_head = mlp(r_dim, d_model, 3, dropout)

    # ------------------------------------------------------------------ pieces
    def encode(self, x, mask, dom_q, script):
        h = self.inp(self.tok_std(x))
        prefix = self.script(script)
        if self.use_domain:
            prefix = prefix + self.dom(self.q_std(dom_q))
        h = torch.cat([prefix[:, None], h], 1)
        mask = torch.cat([mask.new_ones(mask.shape[0], 1), mask], 1)
        tokens, state = self.stack(h, mask)
        return tokens, mask, state

    def read(self, tokens, mask, state, aq):
        """aq: (B, A, enc_dim) -> r (B, A, 2*inner), mag features (B, A, 2H+1)."""
        B, A, _ = aq.shape
        q = self.query(self.q_std(aq)).view(B, A, 2, self.heads, -1)
        if self.tied:
            kf, kb = self.stack.key_of(self.inp(self.tok_std(aq)).flatten(0, 1))
            q = q + torch.stack([kf, kb], 1).view(B, A, 2, self.heads, -1)
        (sf, sb) = state
        rf, mf = mlstm_read(sf, q[:, :, 0])
        rb, mb = mlstm_read(sb, q[:, :, 1])
        r = torch.cat([rf.flatten(2), rb.flatten(2)], -1)
        rp = None
        if self.readout in ("meanpool", "hybrid"):
            pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            rp = self.pool_proj(torch.cat([pooled[:, None].expand(B, A, -1), q.flatten(2)], -1))
            if self.readout == "meanpool":
                r, rp = rp, None
            else:
                rp = self.p_norm(rp)
        log_len = torch.log(mask.sum(1).float())[:, None, None].expand(B, A, 1)
        mag = torch.cat([mf, mb, log_len], -1) / 10.0
        return self.r_norm(r), mag, rp

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        x, mask, aq, a_mask = batch["x"], batch["mask"], batch["aq"], batch["a_mask"]
        tokens, mask, state = self.encode(x, mask, batch["dom_q"], batch["script"])
        B = x.shape[0]
        allq = torch.cat([aq, self.overall_q.expand(B, 1, -1)], 1)
        r, mag, rp = self.read(tokens, mask, state, allq)
        r_asp, r_all, mag = r[:, :-1], r[:, -1], mag[:, :-1]

        sent = self.sent_head(r_asp)
        pres_in = {"mag+value": torch.cat([mag, r_asp], -1), "mag": mag, "value": r_asp}[self.presence_mode]
        if rp is not None:
            pres_in = torch.cat([pres_in, rp[:, :-1]], -1)
        pres = self.pres_head(pres_in).squeeze(-1)
        d_overall = 0.0
        if self.direct != "off":
            xs, m0 = (x.float() if self.direct_raw else self.tok_std(x)), batch["mask"].float()[..., None]
            g = self.g(torch.cat([xs[:, 0], (xs * m0).sum(1) / m0.sum(1).clamp(min=1)], -1))
            A0 = aq.shape[1]
            ids = torch.zeros(B, A0, dtype=torch.long)
            for i, names in enumerate(batch["aspect_names"]):
                ids[i, :len(names)] = torch.tensor([self.aspect_vocab.get(n, 0) for n in names], dtype=torch.long)
            ids = ids.to(x.device)
            if self.training and self.id_drop > 0:    # keep the name-generated weights useful (zero-shot)
                ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.id_drop, 0)
            qs = self.q_std(aq)
            dp, ds = self.d_pres(g, qs, ids).squeeze(-1), self.d_sent(g, qs, ids)
            if self.direct == "only":
                pres, sent = dp, ds
            else:
                pres, sent = pres + dp, sent + ds
            d_overall = self.d_overall(g)
        no_neu = ~batch["allow_neutral"].to(sent.device)
        sent = sent.masked_fill(no_neu[:, None, None] & (torch.arange(3, device=sent.device) == 1), -1e4)

        # overall = learned aggregation of aspect sentiments + direct "overall opinion" read
        A = aq.shape[1]
        dom = self.dom(self.q_std(batch["dom_q"])) if self.use_domain else torch.zeros_like(tokens[:, 0])
        aq_p = self.asp_proj(self.q_std(aq))
        imp = F.softplus(self.importance(torch.cat([aq_p, dom[:, None].expand(B, A, -1)], -1))).squeeze(-1)
        w = imp * torch.sigmoid(pres) * a_mask
        P = sent.softmax(-1)
        feats = torch.stack([(w * P[..., 0]).sum(1), (w * P[..., 2]).sum(1), (w * P[..., 1]).sum(1), w.sum(1)], -1)
        overall = self.agg(feats) + (self.overall_head(r_all) if self.direct != "only" else 0.0) + d_overall
        return {"presence": pres, "sentiment": sent, "overall": overall, "importance": imp}
