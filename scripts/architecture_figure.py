"""Architecture figure of QuAM-GDN -> paper_journal/figures/architecture.{pdf,png} (and paper/figures).

    python scripts/architecture_figure.py
"""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
C = {"input": "#eef1f4", "enc": "#dfe7f2", "query": "#e3f1ea", "mem": "#fbe9d0", "direct": "#ece6f5",
     "head": "#d9efe4", "out": "#f3f3f3", "edge": "#3a4148", "accent": "#b0661a", "text": "#1d2328"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 150); ax.set_ylim(-1, 82); ax.axis("off")


def box(x, y, w, h, text, color, title=None, bold=False, lw=1.2, ec=None, fs=9.5, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color,
                                ec=ec or C["edge"], lw=lw, ls="--" if dashed else "-"))
    if title:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=fs + 0.8, weight="bold", color=C["text"])
        ax.text(x + w / 2, y + (h - 4.8) / 2 + 0.6, text, ha="center", va="center", fontsize=fs, color=C["text"], linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["text"],
                weight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, text=None, rad=0.0, color=None, off=(0, 1.3), ha="center", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=color or C["edge"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha, va="bottom", fontsize=8.3,
                color="#4a5259", style="italic")


# ---- title
ax.text(75, 80.5, "QuAM-GDN: aspect-prompted encoder with a bidirectional Gated DeltaNet memory",
        ha="center", va="top", fontsize=13.5, weight="bold", color=C["text"])

# ---- 1 input
box(3, 64, 144, 9.5, "", C["input"])
ax.text(5, 70.8, "① Input (one pass per review)", fontsize=10, weight="bold", color=C["text"], va="center")
ax.text(75, 66.6, "<s>   Mobile  :  battery  |  delivery  |  price  |  camera  |  …   </s></s>   Battery valo but delivery late.   </s>",
        ha="center", va="center", family="DejaVu Sans Mono", fontsize=10.5, color=C["text"])
ax.text(40, 62.0, "category + all candidate aspect names (≈13 names, ≈41 tokens)", ha="center", fontsize=8.3,
        color="#4a5259", style="italic")
ax.text(118, 62.0, "review (Bangla / Banglish / English)", ha="center", fontsize=8.3, color="#4a5259", style="italic")

# ---- 2 encoder
box(3, 47, 144, 10, "fine-tuned XLM-R base (BanglaBERT for BABSA)  ·  12 layers, d = 768  ·  frozen word embeddings, gradient checkpointing\n"
    "names and review attend to each other  →  contextual token states  h₁ … hₙ", C["enc"],
    title="② Fine-tuned multilingual encoder")
arrow((75, 63.5), (75, 57.6))

# ---- 3 three branches
q = box(3, 23, 40, 18, "mean of hᵢ over each aspect's own name span\nq_a = LN( mean(h_span(a)) )\n\nunseen names → same procedure,\nno new parameters",
        C["query"], title="③ Aspect queries")
m = box(50, 23, 52, 18, "project 768 → 256, prepend [category + script] token\n2 blocks × (forward + backward) Gated DeltaNet\n4 heads × 128,  SiLU output gate\nkeep final states  S→_T , S←_T   (128 × 128 per head)",
        C["mem"], title="④ Bidirectional Gated DeltaNet memory", ec=C["accent"], lw=2.0)
d = box(109, 23, 38, 18, "g = MLP( LN[ h<s> ; mean h ] )\n\ns_a = e_id(a) · g  +  MLP[ g ; q_a ]\n(q_a from ③; learned row per known\naspect, unseen aspect → name part only)",
        C["direct"], title="⑤ Direct per-aspect scorer")
arrow((23, 46.6), (23, 41.6), "name spans", off=(1, 0), ha="left")
arrow((76, 46.6), (76, 41.6), "review tokens", off=(1, 0), ha="left")
arrow((128, 46.6), (128, 41.6), "<s> + mean pool", off=(1, 0), ha="left")

# ---- read
r = box(50, 4.5, 52, 13.5, "query MLP:  q_a → q̂_a  (per head, per direction)\nr_a = S_Tᵀ q̂_a      μ_a = log(1 + ‖r_a‖)  (mention signal)\npresence = MLP[μ_a ; r_a ; pooled]\nsentiment = MLP(LN r_a)",
        C["head"], title="⑥ Read memory with the aspect name", fs=9)
arrow((76, 22.6), (76, 18.6), "S_T", off=(1.2, -1.2), ha="left")
arrow((43.6, 28), (49.6, 13), "q_a", rad=-0.25, off=(-2.6, 0), ha="right")

# ---- combine + outputs
plus = (110.5, 11.2)
ax.add_patch(plt.Circle(plus, 2.4, fc="white", ec=C["edge"], lw=1.4))
ax.text(*plus, "+", ha="center", va="center", fontsize=15, weight="bold")
arrow((102.4, 11.2), (108.0, 11.2))
arrow((112, 22.6), (110.8, 13.7))
o = box(117, 0.5, 30, 20.5, "per aspect a:\n• mentioned? (p_a)\n• positive / negative\n  (+ neutral on BABSA)\n\nper review:\n• overall sentiment",
        C["out"], title="⑦ Outputs")
arrow((113.0, 11.2), (116.5, 11.2))

# ---- GDN inset
ax.add_patch(FancyBboxPatch((2, 0.5), 43, 18, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=C["accent"], lw=1.2, ls="--"))
ax.text(23.5, 16.6, "Gated delta rule (per head, per direction)", ha="center", va="top", fontsize=9.5, weight="bold", color=C["accent"])
ax.text(23.5, 10.8, r"$S_t=\alpha_t S_{t-1}+\beta_t\,k_t\,(v_t-\alpha_t S_{t-1}^{\top}k_t)^{\top}$", ha="center", va="center", fontsize=11.5)
ax.text(23.5, 4.6, "remove what the memory returns for $k_t$, then write the correction:\nrepeated / contradicting mentions overwrite (xLSTM only adds).\n"
        r"$\alpha_t$: forget gate (init. slow, 0.95–0.998)   $\beta_t$: write strength", ha="center", va="center", fontsize=7.6,
        color="#4a5259", linespacing=1.35)

for d_ in (ROOT / "paper_journal" / "figures", ROOT / "paper" / "figures"):
    d_.mkdir(parents=True, exist_ok=True)
out = ROOT / "paper_journal" / "figures"
fig.savefig(out / "architecture.pdf", bbox_inches="tight")
fig.savefig(out / "architecture.png", dpi=220, bbox_inches="tight")
shutil.copy(out / "architecture.pdf", ROOT / "paper" / "figures" / "architecture.pdf")
print("saved", out / "architecture.png")
