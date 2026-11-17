"""Minimal bidirectional xLSTM (mLSTM + sLSTM) in plain PyTorch.

Follows Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024), with one
addition QuAM needs: the mLSTM returns its final matrix memory (C_T, n_T, m_T) so that it
can be queried afterwards (`mlstm_read`).

Stabilised mLSTM (per head, k scaled by 1/sqrt(d)):
    m_t = max(log f_t + m_{t-1}, i~_t)
    i'_t = exp(i~_t - m_t),  f'_t = exp(log f_t + m_{t-1} - m_t)
    C_t = f'_t C_{t-1} + i'_t v_t k_t^T,   n_t = f'_t n_{t-1} + i'_t k_t
    h_t = C_t q_t / max(|n_t^T q_t|, exp(-m_t))
C and n are stored scaled by exp(-m); the true memory is exp(m) * C.
Right padding is handled by making padded steps no-ops (i~ = -inf, log f = 0), so the
state at the last position equals the state after the last real token.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def _f(x):
    """Upcast to at least float32 (keeps float64 for tests)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


# --------------------------------------------------------------------------- mLSTM core
def _masked_gates(igate, logf, mask):
    if mask is None:
        return igate, logf
    valid = mask[:, None, :]
    return igate.masked_fill(~valid, NEG_INF), logf.masked_fill(~valid, 0.0)


GATE_CAP = 15.0


def softcap(x, cap):
    """cap * tanh(x / cap): bounds gate pre-activations (as in xLSTM 7B) so the
    stabiliser m stays in [-cap, cap] and exp(-m) cannot overflow."""
    return x if cap is None else cap * torch.tanh(x / cap)


def fp32(fn):
    """Run fn with autocast disabled: the exponential gating needs full precision."""
    def wrapped(*a, **kw):
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            return fn(*a, **kw)
    wrapped.__doc__, wrapped.__name__ = fn.__doc__, fn.__name__
    return wrapped


@fp32
def mlstm_parallel(q, k, v, igate, fgate, mask=None, eps=1e-6, cap=GATE_CAP):
    """q, k, v: (B, H, T, D); igate, fgate: (B, H, T) pre-activations; mask: (B, T) bool.

    Returns h (B, H, T, D) and the final state (C (B,H,D,D), n (B,H,D), m (B,H)).
    """
    B, H, T, D = q.shape
    k = k / math.sqrt(D)
    igate, logf = _masked_gates(softcap(_f(igate), cap), F.logsigmoid(softcap(_f(fgate), cap)), mask)
    cum = torch.cumsum(logf, dim=-1)
    logd = cum[..., :, None] - cum[..., None, :] + igate[..., None, :]          # (B,H,T,T)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    logd = logd.masked_fill(~causal, NEG_INF)
    m = logd.amax(dim=-1)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    decay = torch.exp(logd - m[..., None])
    qf, kf, vf = _f(q), _f(k), _f(v)
    s = (qf @ kf.transpose(-1, -2)) * decay
    norm = torch.maximum(s.sum(-1).abs(), torch.exp(-m))
    h = (s @ vf) / (norm[..., None] + eps)
    w = decay[..., -1, :]                                                        # (B,H,T)
    C = torch.einsum("bhs,bhsd,bhse->bhde", w, vf, kf)
    n = torch.einsum("bhs,bhse->bhe", w, kf)
    return h.to(q.dtype), (C, n, m[..., -1])


def mlstm_recurrent(q, k, v, igate, fgate, mask=None, eps=1e-6, cap=GATE_CAP):
    """Step-by-step reference implementation (used in tests)."""
    B, H, T, D = q.shape
    k = k / math.sqrt(D)
    igate, logf = _masked_gates(softcap(_f(igate), cap), F.logsigmoid(softcap(_f(fgate), cap)), mask)
    C = q.new_zeros(B, H, D, D, dtype=_f(q).dtype)
    n = q.new_zeros(B, H, D, dtype=_f(q).dtype)
    m = torch.full((B, H), NEG_INF, device=q.device, dtype=_f(q).dtype)
    hs = []
    for t in range(T):
        m_new = torch.maximum(logf[..., t] + m, igate[..., t])
        m_new = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new) if t == 0 else m_new
        i_ = torch.exp(igate[..., t] - m_new)
        f_ = torch.exp(logf[..., t] + m - m_new)
        vt, kt, qt = _f(v[..., t, :]), _f(k[..., t, :]), _f(q[..., t, :])
        C = f_[..., None, None] * C + i_[..., None, None] * (vt[..., :, None] * kt[..., None, :])
        n = f_[..., None] * n + i_[..., None] * kt
        num = (C @ qt[..., None]).squeeze(-1)
        den = torch.maximum((n * qt).sum(-1).abs(), torch.exp(-m_new))
        hs.append(num / (den[..., None] + eps))
        m = m_new
    return torch.stack(hs, dim=2).to(q.dtype), (C, n, m)


@fp32
def mlstm_read(state, query, eps=1e-6):
    """Query a final mLSTM memory. state: (C, n, m); query: (B, A, H, D).

    Returns
      r       (B, A, H, D)  retrieved value  C q / max(|n^T q|, exp(-m))
      log_mag (B, A, H)     log |n_true^T q| = log |n^T q| + m  -- how strongly keys matching q
                            were written into memory (QuAM's presence signal)
    """
    C, n, m = state
    q = _f(query).to(state[0].dtype)
    num = torch.einsum("bhde,bahe->bahd", C, q)
    dot = torch.einsum("bhe,bahe->bah", n, q)
    den = torch.maximum(dot.abs(), torch.exp(-m)[:, None, :])
    r = num / (den[..., None] + eps)
    # log(|n^T q| + e^-m) + m = log(|n_true^T q| + 1): a smooth version of the paper's
    # max(|n^T q|, 1) normaliser. Bounded gradient (log|x| alone explodes as x -> 0).
    log_mag = torch.log(dot.abs() + torch.exp(-m)[:, None, :]) + m[:, None, :]
    return r, log_mag


# --------------------------------------------------------------------------- helpers
def reverse_padded(x, mask):
    """Reverse each sequence's valid prefix; padding stays at the end. x: (B, T, ...)."""
    B, T = mask.shape
    lengths = mask.sum(1, keepdim=True)
    t = torch.arange(T, device=x.device)[None, :]
    idx = torch.where(t < lengths, lengths - 1 - t, t)
    return x.gather(1, idx.view(B, T, *([1] * (x.dim() - 2))).expand_as(x))


class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim)

    def forward(self, x):                       # (B, T, D)
        x = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(x).transpose(1, 2)


class MultiHeadNorm(nn.Module):
    """Per-head LayerNorm (the xLSTM paper's GroupNorm over heads)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):                       # (..., dim)
        sh = x.shape
        x = x.view(*sh[:-1], self.heads, -1)
        x = F.layer_norm(x, x.shape[-1:])
        return x.reshape(sh) * self.weight


# --------------------------------------------------------------------------- mLSTM block
class MLSTMCell(nn.Module):
    """q/k from a causal conv branch, v from the input; gates from [q, k, v]."""

    def __init__(self, dim, heads, conv_kernel=4):
        super().__init__()
        self.heads = heads
        self.conv = CausalConv1d(dim, conv_kernel)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.igate = nn.Linear(3 * dim, heads)
        self.fgate = nn.Linear(3 * dim, heads)
        nn.init.zeros_(self.igate.weight); nn.init.normal_(self.igate.bias, std=0.1)
        nn.init.zeros_(self.fgate.weight)
        with torch.no_grad():
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, heads))
        self.norm = MultiHeadNorm(dim, heads)
        self.skip = nn.Parameter(torch.ones(dim))

    def forward(self, x, mask):                 # x: (B, T, dim)
        B, T, D = x.shape
        xc = F.silu(self.conv(x))
        q, k, v = self.q(xc), self.k(xc), self.v(x)
        g = torch.cat([q, k, v], -1)
        ig, fg = self.igate(g), self.fgate(g)   # (B, T, H)
        split = lambda t: t.view(B, T, self.heads, -1).transpose(1, 2)
        h, state = mlstm_parallel(split(q), split(k), split(v), ig.transpose(1, 2), fg.transpose(1, 2), mask)
        h = h.transpose(1, 2).reshape(B, T, D)
        return self.norm(h) + self.skip * xc, state


class BiMLSTMBlock(nn.Module):
    """Pre-norm residual mLSTM block (up-projection 2x), run forward and backward."""

    def __init__(self, d_model, heads, proj_factor=2, dropout=0.0):
        super().__init__()
        inner = d_model * proj_factor
        self.inner, self.heads = inner, heads
        self.ln = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 2 * inner)
        self.fwd, self.bwd = MLSTMCell(inner, heads), MLSTMCell(inner, heads)
        self.down = nn.Linear(inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        a, z = self.up(self.ln(x)).chunk(2, -1)
        hf, sf = self.fwd(a, mask)
        hb, sb = self.bwd(reverse_padded(a, mask), mask)
        h = (hf + reverse_padded(hb, mask)) * F.silu(z)
        return x + self.drop(self.down(h)), (sf, sb)


# --------------------------------------------------------------------------- sLSTM block
class SLSTMCell(nn.Module):
    """Scalar-memory LSTM with exponential input gate and block-diagonal (per-head) memory mixing."""

    def __init__(self, dim, heads, conv_kernel=4):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.conv = CausalConv1d(dim, conv_kernel)
        self.w_if = nn.Linear(dim, 2 * dim)      # from conv branch
        self.w_zo = nn.Linear(dim, 2 * dim)      # from raw input
        self.r = nn.Parameter(torch.randn(heads, self.dh, 4 * self.dh) / math.sqrt(self.dh))
        with torch.no_grad():
            self.w_if.bias[dim:].copy_(torch.linspace(3.0, 6.0, dim))  # forget-gate bias
        self.norm = MultiHeadNorm(dim, heads)

    def forward(self, x, mask):
        B, T, D = x.shape
        H, dh = self.heads, self.dh
        xi, xf = self.w_if(F.silu(self.conv(x))).float().chunk(2, -1)
        xz, xo = self.w_zo(x).float().chunk(2, -1)
        pre = torch.stack([xi, xf, xz, xo], 2).view(B, T, 4, H, dh)
        h = x.new_zeros(B, H, dh, dtype=torch.float)
        c, n = torch.zeros_like(h), torch.zeros_like(h)
        m = torch.full_like(h, -1e30)
        r = self.r.float()
        outs = []
        for t in range(T):
            rec = torch.einsum("bhd,hde->bhe", h, r).view(B, H, 4, dh).transpose(1, 2)
            i_t, f_t, z_t, o_t = (pre[:, t] + rec).unbind(1)
            i_t, f_t = softcap(i_t, GATE_CAP), softcap(f_t, GATE_CAP)
            logf = F.logsigmoid(f_t)
            m_new = torch.maximum(logf + m, i_t)
            i_ = torch.exp(i_t - m_new)
            f_ = torch.exp(logf + m - m_new)
            c_new = f_ * c + i_ * torch.tanh(z_t)
            n_new = f_ * n + i_
            h_new = torch.sigmoid(o_t) * c_new / n_new.clamp(min=1e-6)
            keep = mask[:, t].view(B, 1, 1)
            c, n, m, h = (torch.where(keep, a, b) for a, b in ((c_new, c), (n_new, n), (m_new, m), (h_new, h)))
            outs.append(h_new)
        out = torch.stack(outs, 1).reshape(B, T, D).to(x.dtype)
        return self.norm(out)


class BiSLSTMBlock(nn.Module):
    """Post-up-projection sLSTM block: sLSTM (both directions) then a gated feed-forward."""

    def __init__(self, d_model, heads, ff_factor=4 / 3, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.fwd, self.bwd = SLSTMCell(d_model, heads), SLSTMCell(d_model, heads)
        ff = int(round(d_model * ff_factor / 64)) * 64
        self.ff_up = nn.Linear(d_model, 2 * ff)
        self.ff_down = nn.Linear(ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.ln1(x)
        h = self.fwd(y, mask) + reverse_padded(self.bwd(reverse_padded(y, mask), mask), mask)
        x = x + self.drop(h)
        a, g = self.ff_up(self.ln2(x)).chunk(2, -1)
        return x + self.drop(self.ff_down(a * F.gelu(g))), None


class BiXLSTMStack(nn.Module):
    """layout: string of 'm'/'s', e.g. 'mmsm'. The last block must be 'm' (its memory is queried)."""

    def __init__(self, d_model, heads, layout="mmsm", dropout=0.0):
        super().__init__()
        assert layout[-1] == "m", "QuAM reads the memory of the last block, which must be mLSTM"
        self.blocks = nn.ModuleList(
            BiMLSTMBlock(d_model, heads, dropout=dropout) if c == "m" else BiSLSTMBlock(d_model, heads, dropout=dropout)
            for c in layout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        state = None
        for blk in self.blocks:
            x, s = blk(x, mask)
            state = s if s is not None else state
        return self.norm(x), state   # state: (fwd (C,n,m), bwd (C,n,m)) of the last block
