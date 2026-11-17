"""Diagnostics for the presence-learning problem: query anisotropy and memory-read statistics."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quam.data import OVERALL_QUERY, ABSADataset, EncoderCache, collate, default_cache_dir, load_records  # noqa: E402
from quam.model import QuAM  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
cache = EncoderCache(default_cache_dir())
recs = load_records(("eco",))
names = sorted({a["query"] for r in recs for a in r["aspects"]})
E = torch.stack([cache.query(n) for n in names])


def cos_stats(X, label):
    Xn = F.normalize(X, dim=-1)
    c = Xn @ Xn.T
    off = c[~torch.eye(len(X), dtype=torch.bool)]
    print(f"{label:34} mean cos={off.mean():.3f}  min={off.min():.3f}  max={off.max():.3f}")


cos_stats(E, "aspect queries (raw XLM-R)")
cos_stats(F.layer_norm(E, E.shape[-1:]), "aspect queries after LayerNorm")
cos_stats(E - E.mean(0), "aspect queries mean-centred")
tok = torch.cat([cache.text(r["text"]) for r in recs[:200]]).float()
cos_stats(tok[torch.randperm(len(tok))[:500]], "review tokens (raw XLM-R)")
print("token feature with largest |mean|:", tok.mean(0).abs().max().item(), " typical std:", tok.std(0).median().item())

torch.manual_seed(0)
ds = ABSADataset(recs, cache, "official", "train", limit=32)
batch = collate([ds[i] for i in range(32)])
model = QuAM(cache.dim, cache.query(OVERALL_QUERY), cache.stats, layout="mm").eval()
with torch.no_grad():
    tokens, mask, (sf, sb) = model.encode(batch["x"], batch["mask"], batch["dom_q"], batch["script"])
    C, n, m = sf
    q = model.query(batch["aq"]).view(32, batch["aq"].shape[1], 2, 4, -1)[:, :, 0]
    dot = torch.einsum("bhe,bahe->bah", n, q)
    floor = torch.exp(-m)[:, None, :].expand_as(dot)
    print(f"\nmemory stabiliser m: mean={m.mean():.2f} min={m.min():.2f} max={m.max():.2f}")
    print(f"|n^T q|: median={dot.abs().median():.3e}  | exp(-m): median={floor.median():.3e}")
    print(f"share of reads where |n^T q| < exp(-m) (denominator floored): {(dot.abs() < floor).float().mean():.2%}")
    print(f"share of reads with |n^T q| < 1e-3 * |n||q| (near-zero denominator): "
          f"{(dot.abs() < 1e-3 * n.norm(dim=-1)[:, None] * q.norm(dim=-1)).float().mean():.2%}")
    r, mag, _ = model.read(tokens, mask, (sf, sb), batch["aq"])
    raw_r = torch.einsum("bhde,bahe->bahd", C, q) / torch.maximum(dot.abs(), floor)[..., None]
    print(f"raw read norm: median={raw_r.norm(dim=-1).median():.2f}  max={raw_r.norm(dim=-1).max():.2e}")
    am = batch["a_mask"]
    within = r[am].view(-1, r.shape[-1])
    per_review_var = torch.stack([r[i][am[i]].var(0).mean() for i in range(32)]).mean()
    print(f"read variance across aspects of one review: {per_review_var:.4f}  vs across everything: {within.var(0).mean():.4f}")
