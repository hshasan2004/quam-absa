"""Bidirectional Gated DeltaNet memory in plain PyTorch, a drop-in alternative to the mLSTM memory.

Gated delta rule (Yang, Kautz & Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta
Rule", ICLR 2025), per head, with L2-normalised keys/queries, S of shape (Dk, Dv):
    S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t k_t v_t^T,   o_t = S_t^T q_t
a_t in (0, 1] is a data-dependent decay (forgetting), b_t in (0, 1) the write strength. Unlike
the mLSTM's purely additive C_t = f C + i v k^T, the delta rule first removes what the memory
currently returns for k_t (S^T k_t) and writes the correction, so a repeated or contradicting
mention of an aspect overwrites the old association instead of piling up on it.

`gdn_chunk` is the chunkwise-parallel form (WY representation, as in the paper / flash-linear-
attention reference), `gdn_recurrent` the step-by-step definition used to test it. Right padding:
padded steps have b = 0 and log a = 0, so they leave S unchanged and S_T is the state after the
last real token. `gdn_read` queries the final memory with aspect queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xlstm import CausalConv1d, MultiHeadNorm, fp32, reverse_padded


def _prep(q, k, beta, logα, mask):
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    beta, logα = beta.float(), logα.float()
    if mask is not None:
        valid = mask[:, None, :]
        beta, logα = beta.masked_fill(~valid, 0.0), logα.masked_fill(~valid, 0.0)
    return q, k, beta, logα


def gdn_recurrent(q, k, v, beta, logα, mask=None):
    """q, k: (B,H,T,Dk), v: (B,H,T,Dv), beta/logα: (B,H,T). Returns o (B,H,T,Dv), S (B,H,Dk,Dv)."""
    q, k, beta, logα = _prep(q, k, beta, logα, mask)
    v = v.float()
    B, H, T, Dk = k.shape
    S = k.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(T):
        S = S * logα[..., t].exp()[..., None, None]
        kt = k[:, :, t]
        u = beta[..., t, None] * (v[:, :, t] - torch.einsum("bhkv,bhk->bhv", S, kt))
        S = S + kt[..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, :, t]))
    return torch.stack(out, 2), S


@fp32
def gdn_chunk(q, k, v, beta, logα, mask=None, chunk=64):
    """Chunkwise-parallel gated delta rule; same inputs/outputs as gdn_recurrent."""
    q, k, beta, logα = _prep(q, k, beta, logα, mask)
    v = v.float()
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = min(chunk, T)
    pad = (-T) % C
    if pad:   # padded steps: no write, no decay
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, logα = F.pad(beta, (0, pad)), F.pad(logα, (0, pad))
    N = (T + pad) // C
    q, k, v = (x.view(B, H, N, C, -1) for x in (q, k, v))
    beta, g = beta.view(B, H, N, C), logα.view(B, H, N, C).cumsum(-1)   # g: within-chunk cumulative log decay
    kb, vb = k * beta[..., None], v * beta[..., None]
    decay = (g[..., :, None] - g[..., None, :])
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strict = tril & ~torch.eye(C, dtype=torch.bool, device=q.device)
    decay_l = decay.masked_fill(~tril, float("-inf")).exp()
    # WY representation: T_mat = (I + strictly-lower(b_i a_i/a_j k_i.k_j))^-1
    A = ((kb @ k.transpose(-1, -2)) * decay_l).masked_fill(~strict, 0.0)
    eye = torch.eye(C, device=q.device).expand_as(A)
    Tm = torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)
    U = Tm @ vb                                  # (B,H,N,C,Dv)
    W = Tm @ (kb * g.exp()[..., None])           # (B,H,N,C,Dk)
    S = k.new_zeros(B, H, Dk, Dv)
    outs = []
    for n in range(N):
        qn, kn, gn = q[:, :, n], k[:, :, n], g[:, :, n]
        vnew = U[:, :, n] - W[:, :, n] @ S                                  # corrected values
        attn = ((qn @ kn.transpose(-1, -2)) * decay_l[:, :, n]).masked_fill(~tril, 0.0)
        outs.append((qn * gn.exp()[..., None]) @ S + attn @ vnew)
        glast = gn[..., -1]
        S = S * glast.exp()[..., None, None] + (kn * (glast[..., None] - gn).exp()[..., None]).transpose(-1, -2) @ vnew
    o = torch.stack(outs, 2).reshape(B, H, N * C, Dv)[:, :, :T]
    return o, S


@fp32
def gdn_read(S, query, eps=1e-6):
    """Query a final GDN memory. S: (B,H,Dk,Dv); query: (B,A,H,Dk).

    Returns r (B,A,H,Dv) = S^T q (q L2-normalised) and log_mag (B,A,H) = log(1 + |r|): how much
    content the memory holds under that key (the presence signal, like the mLSTM's log|n^T q|).
    """
    q = F.normalize(query.float(), dim=-1)
    r = torch.einsum("bhkv,bahk->bahv", S.float(), q)
    return r, torch.log1p(r.norm(dim=-1) + eps)


class GDNCell(nn.Module):
    """q/k from a causal conv branch (SiLU), v from the input; write strength b and decay a per head."""

    def __init__(self, dim, heads, conv_kernel=4):
        super().__init__()
        self.heads = heads
        self.conv = CausalConv1d(dim, conv_kernel)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.b = nn.Linear(dim, heads)
        self.a = nn.Linear(dim, heads)
        nn.init.zeros_(self.a.weight)
        with torch.no_grad():
            self.a.bias.copy_(torch.linspace(3.0, 6.0, heads))   # slow forgetting at init, as the mLSTM
        self.norm = MultiHeadNorm(dim, heads)
        self.skip = nn.Parameter(torch.ones(dim))

    def forward(self, x, mask):
        B, T, D = x.shape
        xc = F.silu(self.conv(x))
        split = lambda t: t.view(B, T, self.heads, -1).transpose(1, 2)
        q, k, v = split(self.q(xc)), split(self.k(xc)), split(self.v(x))
        beta = torch.sigmoid(self.b(x)).transpose(1, 2)
        logα = F.logsigmoid(self.a(x).float()).transpose(1, 2)
        o, S = gdn_chunk(q, k, v, beta, logα, mask)
        o = o.to(x.dtype).transpose(1, 2).reshape(B, T, D)
        return self.norm(o) + self.skip * xc, S


class BiGDNBlock(nn.Module):
    """Pre-norm residual Gated DeltaNet block (up-projection 2x, SiLU output gate), both directions."""

    def __init__(self, d_model, heads, proj_factor=2, dropout=0.0):
        super().__init__()
        inner = d_model * proj_factor
        self.inner, self.heads = inner, heads
        self.ln = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 2 * inner)
        self.fwd, self.bwd = GDNCell(inner, heads), GDNCell(inner, heads)
        self.down = nn.Linear(inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        a, z = self.up(self.ln(x)).chunk(2, -1)
        hf, sf = self.fwd(a, mask)
        hb, sb = self.bwd(reverse_padded(a, mask), mask)
        h = (hf + reverse_padded(hb, mask)) * F.silu(z)
        return x + self.drop(self.down(h)), (sf, sb)
