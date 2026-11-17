"""Numerical checks for quam/xlstm.py. Run: python -m pytest tests -q  (or python tests/test_xlstm.py)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quam.xlstm import (BiXLSTMStack, mlstm_parallel, mlstm_read, mlstm_recurrent,  # noqa: E402
                        reverse_padded)

torch.manual_seed(0)
B, H, T, D = 3, 2, 11, 8


def rand_inputs(scale=3.0):
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    ig = torch.randn(B, H, T, dtype=torch.float64) * scale
    fg = torch.randn(B, H, T, dtype=torch.float64) * scale
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = False
    mask[2, 3:] = False
    return q, k, v, ig, fg, mask


def close(a, b, tol=1e-8):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def test_parallel_equals_recurrent():
    q, k, v, ig, fg, mask = rand_inputs()
    hp, (Cp, np_, mp) = mlstm_parallel(q, k, v, ig, fg, mask)
    hr, (Cr, nr, mr) = mlstm_recurrent(q, k, v, ig, fg, mask)
    valid = mask[:, None, :, None].expand_as(hp)
    assert close(hp[valid], hr[valid]), (hp - hr).abs()[valid].max()
    assert close(mp, mr) and close(Cp, Cr) and close(np_, nr)


def test_padding_does_not_change_final_state():
    q, k, v, ig, fg, mask = rand_inputs()
    _, (C, n, m) = mlstm_parallel(q, k, v, ig, fg, mask)
    L = int(mask[2].sum())
    _, (C2, n2, m2) = mlstm_parallel(q[2:, :, :L], k[2:, :, :L], v[2:, :, :L], ig[2:, :, :L], fg[2:, :, :L])
    assert close(C[2:], C2) and close(n[2:], n2) and close(m[2:], m2)


def test_read_with_last_query_reproduces_last_output():
    """Reading the final memory with q_T must give exactly the layer's output h_T."""
    q, k, v, ig, fg, _ = rand_inputs()
    h, state = mlstm_parallel(q, k, v, ig, fg)
    r, log_mag = mlstm_read(state, q[:, :, -1][:, None])           # A = 1
    assert close(r[:, 0], h[:, :, -1])
    assert torch.isfinite(log_mag).all()


def test_log_mag_is_stabiliser_free():
    """log_mag = log(|n_true^T q| + 1) must not depend on the stabiliser m.

    Shifting every input gate by c multiplies the true memory by e^c, so
    log_mag must become log(e^c * |n_true^T q| + 1).
    """
    q, k, v, ig, fg, _ = rand_inputs()
    qa = torch.randn(B, 4, H, D, dtype=torch.float64)
    _, s1 = mlstm_parallel(q, k, v, ig, fg, cap=None)
    _, s2 = mlstm_parallel(q, k, v, ig + 5.0, fg, cap=None)
    _, m1 = mlstm_read(s1, qa)
    _, m2 = mlstm_read(s2, qa)
    true_dot = torch.exp(m1) - 1
    assert close(m2, torch.log(torch.exp(torch.tensor(5.0, dtype=torch.float64)) * true_dot + 1), tol=1e-6)


def test_log_mag_gradient_is_bounded():
    """The presence signal must have a finite gradient even when n^T q is exactly 0."""
    C = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    n = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    m = torch.zeros(1, 1, dtype=torch.float64)
    qa = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=torch.float64, requires_grad=True)  # n^T q = 0
    _, log_mag = mlstm_read((C, n, m), qa)
    log_mag.sum().backward()
    assert torch.isfinite(qa.grad).all() and qa.grad.abs().max() <= 1.0 + 1e-9


def test_softcap_bounds_stabiliser():
    """With gate soft-capping, extreme gate values cannot push m outside [-cap, cap]."""
    q, k, v, ig, fg, mask = rand_inputs()
    _, (C, n, m) = mlstm_parallel(q, k, v, ig * 1e4, fg * 1e4, mask)
    assert m.abs().max() <= 15.0 + 1e-9 and torch.isfinite(C).all() and torch.isfinite(n).all()


def test_reverse_padded_roundtrip():
    x = torch.arange(B * T).view(B, T, 1).float()
    _, _, _, _, _, mask = rand_inputs()
    r = reverse_padded(x, mask)
    assert r[2, 0, 0] == x[2, 2, 0] and r[2, 5, 0] == x[2, 5, 0]
    assert torch.equal(reverse_padded(r, mask), x)


def test_stack_runs_and_backprops():
    stack = BiXLSTMStack(32, 4, "msm")
    x = torch.randn(2, 9, 32, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 5:] = False
    y, (sf, sb) = stack(x, mask)
    assert y.shape == x.shape and sf[0].shape == (2, 4, 16, 16)
    y[mask].sum().backward()
    assert torch.isfinite(x.grad).all()
    # padded positions of sequence 1 must not influence its real outputs
    x2 = x.detach().clone()
    x2[1, 5:] = 100.0
    y2, _ = stack(x2, mask)
    assert torch.allclose(y.detach()[1, :5], y2[1, :5], atol=1e-5)


def test_against_official_xlstm():
    """Compare with NX-AI's reference parallel kernel, if the package is installed."""
    try:
        from xlstm.blocks.mlstm.backends import parallel_stabilized_simple
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped: official xlstm not importable: {type(e).__name__})")
        return
    q, k, v, ig, fg, _ = rand_inputs(scale=1.0)
    ours, _ = mlstm_parallel(q, k, v, ig, fg, cap=None)
    ref = parallel_stabilized_simple(q, k, v, ig[..., None], fg[..., None])
    err = (ours - ref).abs().max().item()
    print(f"  max |ours - official| = {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
