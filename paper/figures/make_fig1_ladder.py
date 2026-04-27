"""Generate Figure 1 (paper) — three-panel behavioural ladder.

Renders one panel per architecture (Aurora, MM-TSFlib, TaTS) showing the
relative MSE change vs. C1 across the eight conditions. Two-cluster
colour coding makes the prior-vs-text dichotomy glanceable.

Inputs:  summaries/deep/ladder_table.csv (locked, canonical-only)
Outputs: paper/figures/fig1_ladder.pdf, .png

Run with:
    /home/karthik/miniconda3/envs/aurora/bin/python \
        paper/figures/make_fig1_ladder.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
LADDER = PROJECT / "summaries" / "deep" / "ladder_table.csv"
OUT_DIR = PROJECT / "paper" / "figures"

# Order conditions as they appear in Table 1 of the paper.
COND_ORDER = [
    "C1_original", "C3_shuffled", "C4_crossdomain", "C8_oracle",
    "C9_zero_priors", "C2_empty", "C5_constant", "C6_unimodal",
]
COND_LABELS = {
    "C1_original":     "C1 orig",
    "C3_shuffled":     "C3 shuf",
    "C4_crossdomain":  "C4 cross",
    "C8_oracle":       "C8 oracle",
    "C9_zero_priors":  "C9 zero-priors",
    "C2_empty":        "C2 empty",
    "C5_constant":     "C5 const",
    "C6_unimodal":     "C6 uni",
}
# Two-cluster + unimodal colours, matching Table 1's row colouring.
COLOR_TEXT = "#3870b6"     # blue: text-perturb cluster
COLOR_PRIOR = "#cc4f4f"    # red:  prior-zero cluster
COLOR_UNI = "#5aa970"      # green: architectural unimodal
COLOR_REF = "#888888"      # grey: control

CLUSTER = {
    "C1_original":     ("control",     COLOR_REF),
    "C3_shuffled":     ("text",        COLOR_TEXT),
    "C4_crossdomain":  ("text",        COLOR_TEXT),
    "C8_oracle":       ("text",        COLOR_TEXT),
    "C9_zero_priors":  ("prior",       COLOR_PRIOR),
    "C2_empty":        ("text+prior",  COLOR_PRIOR),
    "C5_constant":     ("text+prior",  COLOR_PRIOR),
    "C6_unimodal":     ("uni",         COLOR_UNI),
}

MODELS = [
    ("aurora", "Aurora (zero-shot)"),
    ("mmtsflib", "MM-TSFlib (late fusion)"),
    ("tats", "TaTS (early fusion)"),
]


def main() -> None:
    df = pd.read_csv(LADDER)
    df = df.set_index("condition").loc[COND_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
    })

    # Per-model annotation strategies (avoid label overlap on
    # near-identical bars in MMTSFlib/TaTS prior-zeroed cluster).
    for ax, (key, title) in zip(axes, MODELS):
        col = f"{key}_pct_vs_C1"
        vals = df[col].values
        labels = [COND_LABELS[c] for c in df.index]
        colors = [CLUSTER[c][1] for c in df.index]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black",
                      linewidth=0.5, width=0.78)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        if key == "aurora":
            ax.set_ylabel(r"$\Delta$MSE (% vs C1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate only the headline bars; skip near-zero bars that visually
        # tell the story themselves.
        for i, (b, v) in enumerate(zip(bars, vals)):
            cond = df.index[i]
            # For trained models (TaTS / MM-TSFlib) we annotate the prior +
            # unimodal cluster (the numerically-meaningful changes).
            if key != "aurora":
                if cond in ("C9_zero_priors", "C2_empty", "C5_constant",
                             "C6_unimodal"):
                    if cond == "C2_empty" and key == "tats":
                        # Three near-identical TaTS bars -> single shared label.
                        ax.annotate(f"+31.1%",
                                    xy=(b.get_x() + b.get_width() / 2, v),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7,
                                    fontweight="bold")
                    elif key == "tats" and cond in ("C9_zero_priors",
                                                     "C5_constant"):
                        pass    # don't double-label
                    elif cond == "C2_empty" and key == "mmtsflib":
                        ax.annotate(f"+2.5%",
                                    xy=(b.get_x() + b.get_width() / 2, v),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7,
                                    fontweight="bold")
                    elif key == "mmtsflib" and cond in ("C9_zero_priors",
                                                          "C5_constant"):
                        pass
                    else:
                        sign = "+" if v >= 0 else ""
                        ax.annotate(f"{sign}{v:.1f}%",
                                    xy=(b.get_x() + b.get_width() / 2, v),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7)
            else:
                # Aurora: annotate every nonzero bar; values are tiny.
                if abs(v) > 1e-4:
                    sign = "+" if v >= 0 else ""
                    ax.annotate(f"{sign}{v:.3f}%",
                                xy=(b.get_x() + b.get_width() / 2, v),
                                xytext=(0, 3 if v >= 0 else -3),
                                textcoords="offset points",
                                ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=6.5)

    # Tight ylims chosen per-panel.
    axes[0].set_ylim(-0.025, 0.06)        # aurora
    axes[1].set_ylim(-0.5, 3.5)           # mmtsflib
    axes[2].set_ylim(-2, 36)              # tats

    fig.suptitle(
        ("Eight-condition ladder: text content moves MSE by <0.06%, "
         "while zeroing the numeric prior moves it by up to +31%"),
        fontsize=8.5, y=1.04,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig1_ladder.pdf"
    out_png = OUT_DIR / "fig1_ladder.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
