"""
Figures for the Omega-S LLM preprint.

figure1_retention renders the per-seed retention results reported in Section 4.4.
It plots every seed, including the one with a negative effect, rather than the
mean alone.

figure2_degree_distribution is NOT generated here: it requires the trained
weight matrices, which are not redistributed with this repository. The script
degree_distribution.py in this folder produces it from a local run.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
BLUE, RED, GREY = "#1f4e79", "#c0504d", "#7f7f7f"

# Per-seed retention, Llama-3-8B + LoRA, CodeAlpaca -> Wikitext-2,
# HumanEval pass@1 (Section 4.4).
seeds = [42, 123, 456]
delta = [2.80, 5.73, -1.84]
mean = float(np.mean(delta))
assert abs(mean - 2.23) < 0.01, "mean must match the value reported in the text"

fig, ax = plt.subplots(figsize=(5.8, 2.9))
y = np.arange(len(seeds))[::-1]

ax.axvline(0, color="0.6", lw=1.0)
ax.axvspan(-2.4, 0, color=RED, alpha=.055, lw=0)

for yi, d in zip(y, delta):
    c = BLUE if d > 0 else RED
    ax.plot([0, d], [yi, yi], color=c, lw=2.2, solid_capstyle="round", alpha=.75)
    ax.plot(d, yi, "o", ms=8, color=c, zorder=3)
    ax.text(d + (0.28 if d > 0 else -0.28), yi, f"{d:+.2f}", va="center",
            ha="left" if d > 0 else "right", fontsize=8.5, color=c)

ax.axvline(mean, color="0.25", lw=1.3, ls="--")
ax.text(mean, -0.85, f"mean {mean:+.2f} pp", ha="center", fontsize=8.5, color="0.25")

ax.set_yticks(y)
ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=8.5)
ax.set_ylim(-1.25, len(seeds) - 0.4)
ax.set_xlim(-2.6, 6.8)
ax.set_xlabel("change in retained HumanEval pass@1 vs. weight decay (pp)")
ax.set_title("Knowledge retention after sequential fine-tuning: every seed shown",
             fontsize=9.5, loc="left")
ax.text(0.995, 0.06, "2 of 3 seeds positive", transform=ax.transAxes,
        ha="right", fontsize=8, color="0.4")

fig.tight_layout()
fig.savefig("figure1_retention.png", bbox_inches="tight")
fig.savefig("figure1_retention.pdf", bbox_inches="tight")
print("figure1_retention written")
