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
                 presence_mode="mag+value", use_domain=True, readout="memory"):
        super().__init__()
        assert presence_mode in PRESENCE_MODES and readout in ("memory", "meanpool")
        self.presence_mode, self.use_domain, self.readout, self.heads = presence_mode, use_domain, readout, heads
        inner = 2 * d_model                    # mLSTM up-projection factor 2
        self.inner = inner
        self.tok_std = Standardize(stats["tok_mean"], stats["tok_std"])
        self.q_std = Standardize(stats["q_mean"], stats["q_std"])
        self.inp = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model), nn.Dropout(dropout))
        self.dom = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model))
        self.script = nn.Embedding(len(SCRIPTS), d_model)
        self.stack = BiXLSTMStack(d_model, heads, layout, dropout)
        self.query = nn.Sequential(nn.LayerNorm(enc_dim), mlp(enc_dim, d_model, 2 * inner))  # fwd + bwd
        self.asp_proj = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d_model))
        self.register_buffer("overall_q", overall_query.float().clone(), persistent=False)

        r_dim = 2 * inner
        self.r_norm = nn.LayerNorm(r_dim)
        if readout == "meanpool":              # ablation: AF-MAT-style pooled tokens + aspect
            self.pool_proj = nn.Linear(d_model + 2 * inner, r_dim)
        self.sent_head = mlp(r_dim, d_model, 3, dropout)
        mag_dim = 2 * heads + 1                # log|n^T q| per head and direction, plus log length
        pres_in = {"mag+value": mag_dim + r_dim, "mag": mag_dim, "value": r_dim}[presence_mode]
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
        (sf, sb) = state
        rf, mf = mlstm_read(sf, q[:, :, 0])
        rb, mb = mlstm_read(sb, q[:, :, 1])
        r = torch.cat([rf.flatten(2), rb.flatten(2)], -1)
        if self.readout == "meanpool":
            pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            r = self.pool_proj(torch.cat([pooled[:, None].expand(B, A, -1), q.flatten(2)], -1))
        log_len = torch.log(mask.sum(1).float())[:, None, None].expand(B, A, 1)
        mag = torch.cat([mf, mb, log_len], -1) / 10.0
        return self.r_norm(r), mag

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        x, mask, aq, a_mask = batch["x"], batch["mask"], batch["aq"], batch["a_mask"]
        tokens, mask, state = self.encode(x, mask, batch["dom_q"], batch["script"])
        B = x.shape[0]
        allq = torch.cat([aq, self.overall_q.expand(B, 1, -1)], 1)
        r, mag = self.read(tokens, mask, state, allq)
        r_asp, r_all, mag = r[:, :-1], r[:, -1], mag[:, :-1]

        sent = self.sent_head(r_asp)
        no_neu = ~batch["allow_neutral"].to(sent.device)
        sent = sent.masked_fill(no_neu[:, None, None] & (torch.arange(3, device=sent.device) == 1), -1e4)
        pres_in = {"mag+value": torch.cat([mag, r_asp], -1), "mag": mag, "value": r_asp}[self.presence_mode]
        pres = self.pres_head(pres_in).squeeze(-1)

        # overall = learned aggregation of aspect sentiments + direct "overall opinion" read
        A = aq.shape[1]
        dom = self.dom(self.q_std(batch["dom_q"])) if self.use_domain else torch.zeros_like(tokens[:, 0])
        aq_p = self.asp_proj(self.q_std(aq))
        imp = F.softplus(self.importance(torch.cat([aq_p, dom[:, None].expand(B, A, -1)], -1))).squeeze(-1)
        w = imp * torch.sigmoid(pres) * a_mask
        P = sent.softmax(-1)
        feats = torch.stack([(w * P[..., 0]).sum(1), (w * P[..., 2]).sum(1), (w * P[..., 1]).sum(1), w.sum(1)], -1)
        overall = self.agg(feats) + self.overall_head(r_all)
        return {"presence": pres, "sentiment": sent, "overall": overall, "importance": imp}
