"""Generate a single explanatory diagram of the seasonal model.

Output: docs/seasonal_model_images/how_the_model_works.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "seasonal_model_images" / "how_the_model_works.png"

C = {
    "bg": "#FBFBFD",
    "panel_train": "#EAF1F8",
    "panel_predict": "#EEF7EE",
    "border_train": "#3D6FA0",
    "border_predict": "#4F9C46",
    "box": "#FFFFFF",
    "box_edge": "#3A3A3A",
    "gk": "#E15759",
    "def": "#4E79A7",
    "mid": "#59A14F",
    "fwd": "#F28E2B",
    "arrow": "#444444",
    "text": "#222222",
    "muted": "#666666",
    "footer": "#F4ECD8",
    "footer_edge": "#B58A2B",
}

POSITIONS = [("Goalkeepers", C["gk"]), ("Defenders", C["def"]),
             ("Midfielders", C["mid"]), ("Forwards", C["fwd"])]


def box(ax, x, y, w, h, *, fc, ec, lw=1.2, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z,
    ))


def arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        color=color or C["arrow"], linewidth=lw, zorder=4,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(50, 97, "How the Seasonal Points Model Learns and Predicts",
            ha="center", fontsize=20, fontweight="bold", color=C["text"])
    ax.text(50, 93.5,
            "Two phases — first the model studies past seasons, then it predicts the one that has not happened yet.",
            ha="center", fontsize=11.5, color=C["muted"])

    # ── PHASE 1: STUDYING (y 50–88) ──────────────────────────────────────────
    box(ax, 2, 50, 96, 38, fc=C["panel_train"], ec=C["border_train"], lw=2.0)
    ax.text(5, 85, "PHASE 1   ·   STUDYING",
            ha="left", fontsize=14, fontweight="bold", color=C["border_train"])
    ax.text(5, 82,
            "We show the model thousands of past players. For each one it sees the stats AND the points they actually scored.",
            ha="left", fontsize=11, color=C["text"])

    # Past seasons card (left)
    box(ax, 5, 55, 20, 22, fc=C["box"], ec=C["box_edge"])
    ax.text(15, 73.5, "PAST SEASONS", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(15, 70.5, "2019-20  →  2023-24", ha="center", fontsize=10, color=C["muted"])
    ax.text(15, 66,
            "≈ 3,800 players,\neach with stats:\nminutes, goals,\nassists, price,\nclean sheets…",
            ha="center", va="center", fontsize=10, color=C["text"])
    ax.text(15, 58.5, "AND  their actual\nfinal points",
            ha="center", va="center", fontsize=10, color=C["border_train"], fontweight="bold")

    # Arrow + label
    arrow(ax, 25.5, 65, 31, 65)
    ax.text(28.2, 67.5, "fed to", ha="center", fontsize=10,
            color=C["muted"], style="italic")

    # Four specialists across the right
    base_x = 31.5
    w = 15
    gap = 2.0
    for i, (label, color) in enumerate(POSITIONS):
        x = base_x + i * (w + gap)
        box(ax, x, 55, w, 22, fc=C["box"], ec=color, lw=2.2)
        ax.text(x + w / 2, 73.2, label, ha="center", fontsize=11,
                fontweight="bold", color=color)
        ax.text(x + w / 2, 70.7, "specialist", ha="center", fontsize=9.5,
                color=C["muted"], style="italic")
        ax.text(x + w / 2, 64,
                "learns the\npatterns that\nturn stats into\npoints — for\n" + label.lower() + " only",
                ha="center", va="center", fontsize=9.5, color=C["text"])

    ax.text(50, 52,
            "Four specialists, because what earns points for a goalkeeper is not what earns points for a forward.",
            ha="center", fontsize=10.5, color=C["muted"], style="italic")

    # ── PHASE 2: PREDICTING (y 8–46) ────────────────────────────────────────
    box(ax, 2, 8, 96, 38, fc=C["panel_predict"], ec=C["border_predict"], lw=2.0)
    ax.text(5, 43, "PHASE 2   ·   PREDICTING",
            ha="left", fontsize=14, fontweight="bold", color=C["border_predict"])
    ax.text(5, 40,
            "Now we hand it a current player — stats only, no answer — and the right specialist returns a season-points prediction.",
            ha="left", fontsize=11, color=C["text"])

    # New player card
    box(ax, 5, 13, 20, 23, fc=C["box"], ec=C["box_edge"])
    ax.text(15, 32.5, "A CURRENT PLAYER", ha="center", fontsize=11, fontweight="bold")
    ax.text(15, 25.5,
            "Position:  MID\nPrice:        £10.5m\nLast yr minutes: 3100\nLast yr goals:     14\nLast yr bonus:   22",
            ha="center", va="center", fontsize=9.8, color=C["text"],
            family="monospace")
    ax.text(15, 16.5, "stats only — no answer",
            ha="center", fontsize=9.5, color=C["border_predict"], style="italic")

    arrow(ax, 25.5, 24, 32, 24)
    ax.text(28.7, 26.5, "routed by\nposition",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Highlighted specialist (MID)
    box(ax, 32.5, 13, 23, 23, fc=C["box"], ec=C["mid"], lw=2.6)
    ax.text(44, 32.5, "Midfielders specialist",
            ha="center", fontsize=11.5, fontweight="bold", color=C["mid"])
    ax.text(44, 27.5,
            "Compares this player's stats\nto the patterns it learned\nfrom past midfielders.",
            ha="center", va="center", fontsize=10, color=C["text"])
    ax.text(44, 19,
            '"Players that looked like this\nin the past scored about…"',
            ha="center", va="center", fontsize=9.5, style="italic", color=C["muted"])

    # Ghosted other specialists
    g_x0 = 58
    for i, (label, color) in enumerate([POSITIONS[0], POSITIONS[1], POSITIONS[3]]):
        x = g_x0 + i * 5.7
        box(ax, x, 17, 5, 14, fc=C["box"], ec=color, lw=1.2, alpha=0.35)
        ax.text(x + 2.5, 24, label[:3], ha="center", va="center",
                fontsize=9, color=color, alpha=0.55, fontweight="bold")
    ax.text(g_x0 + 8, 14.5, "(other specialists, not used here)",
            ha="center", fontsize=9, color=C["muted"], style="italic")

    arrow(ax, 75.5, 24, 81, 24)
    ax.text(78.3, 26.3, "predicts", ha="center", fontsize=10,
            color=C["muted"], style="italic")

    # Prediction output
    box(ax, 81, 13, 16, 23, fc=C["box"], ec=C["border_predict"], lw=2.6)
    ax.text(89, 32.5, "PREDICTED", ha="center", fontsize=11,
            fontweight="bold", color=C["border_predict"])
    ax.text(89, 30, "season points", ha="center", fontsize=10.5,
            color=C["border_predict"])
    ax.text(89, 23, "≈ 168", ha="center", va="center",
            fontsize=30, fontweight="bold", color=C["text"])
    ax.text(89, 16, "an estimate,\nnot a guarantee",
            ha="center", va="center", fontsize=9, color=C["muted"], style="italic")

    # Footer strip
    arrow(ax, 89, 12.5, 89, 7.5)
    box(ax, 18, 1, 64, 5, fc=C["footer"], ec=C["footer_edge"], lw=1.5)
    ax.text(50, 3.5,
            "Repeat for every player  →  squad optimizer picks the best 15 within budget.",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["text"])

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
