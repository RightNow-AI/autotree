"""Animated schematic simulation: AutoTree vs sequential best-of-n serving.

Renders assets/autotree-mechanism.gif. The animation is a schematic of the
execution model, matching the benchmark protocol (sequential best-of-4 vs a
beam-8 tree). The only benchmark number shown is the measured KV-reuse ratio
from docs/first-benchmark.md.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

GREEN = "#76B900"
WHITE = "#e8e8e8"
DIM = "#7a7a7a"
WASTE = "#3a3a3a"
BG = "#0a0a0a"
PANEL = "#131313"

SEG_SEMI = font_manager.FontProperties(family="Segoe UI", weight="semibold")
SEG_REG = font_manager.FontProperties(family="Segoe UI", weight="regular")
MONO = font_manager.FontProperties(family="Consolas")

P = 16              # prompt tokens
C = 12              # completion tokens per chain
NSEQ = 4            # sequential chains (best-of-4)
NTREE = 8           # tree branches (beam-8)
CELL = 0.155
GAP = 0.028
PITCH = CELL + GAP
ROWPITCH = CELL + 2.6 * GAP
COLS = P + C
COLS_PER_FRAME = 0.5
# branch index -> completion column where it is pruned (winner is 0, runner-up 4)
PRUNE_AT = {1: 4, 2: 6, 3: 3, 5: 8, 6: 5, 7: 10}
HOLD = 32
ENDCARD = 60

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12.8)
ax.set_ylim(0, 7.2)
ax.axis("off")


def draw_frame(frame: int) -> None:
    ax.clear()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 12.8, 7.2, facecolor=BG, zorder=0))

    fill_frames = int(COLS / COLS_PER_FRAME)
    end_start = fill_frames + HOLD
    if frame >= end_start:
        draw_endcard(min(1.0, (frame - end_start) / 10))
        return

    progress = min(frame * COLS_PER_FRAME, COLS)
    comp = max(0.0, progress - P)

    ax.text(0.55, 6.80, "One prompt, many candidate answers",
            fontsize=17, color=WHITE, fontproperties=SEG_SEMI)
    ax.text(0.55, 6.44,
            "Schematic simulation of the execution model. Each square is one token of KV cache.",
            fontsize=10.5, color=DIM, fontproperties=SEG_REG)
    ax.text(12.25, 6.80, f"t = {int(progress):02d}", fontsize=12, color=GREEN,
            fontproperties=MONO, ha="right")

    # ------------- left: sequential best-of-4 -------------
    lx = 0.62
    ax.add_patch(Rectangle((lx - 0.28, 1.55), 6.0, 4.45, facecolor=PANEL, zorder=1))
    ax.text(lx, 5.70, "STOCK SERVING", fontsize=13, color=WHITE, fontproperties=SEG_SEMI)
    ax.text(lx + 2.05, 5.70, "sequential best-of-4", fontsize=11, color=DIM,
            fontproperties=SEG_REG)

    top = 5.28
    seq_tokens = 0
    for row in range(NSEQ):
        y = top - row * ROWPITCH
        for col in range(int(progress)):
            x = lx + col * PITCH
            color = (DIM if row == 0 else WASTE) if col < P else WHITE
            ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=color, zorder=3))
            seq_tokens += 1
    if progress > P:
        ax.text(lx + P * PITCH / 2, top - NSEQ * ROWPITCH + 0.02,
                "prompt recomputed and stored 4x", fontsize=9.5, color=DIM,
                fontproperties=SEG_REG, ha="center")

    ax.text(lx, 2.30, f"tokens computed {seq_tokens:4d}", fontsize=11.5,
            color=WHITE, fontproperties=MONO)
    ax.text(lx + 3.1, 2.30, f"KV pages held {seq_tokens:4d}", fontsize=11.5,
            color=WHITE, fontproperties=MONO)
    ax.text(lx, 1.85, "no sharing, no pruning", fontsize=10, color=DIM,
            fontproperties=SEG_REG)

    # ------------- right: AutoTree beam-8 -------------
    rx = 6.95
    ax.add_patch(Rectangle((rx - 0.28, 1.55), 6.0, 4.45, facecolor=PANEL, zorder=1))
    ax.text(rx, 5.70, "AUTOTREE", fontsize=13, color=GREEN, fontproperties=SEG_SEMI)
    ax.text(rx + 1.55, 5.70, "beam-8 tree, one shared prefix", fontsize=11,
            color=DIM, fontproperties=SEG_REG)

    rtop = 5.28
    # shared prefix row
    for col in range(min(int(progress), P)):
        ax.add_patch(Rectangle((rx + col * PITCH, rtop), CELL, CELL,
                               facecolor=GREEN, zorder=3))
    tree_computed = min(int(progress), P)
    physical = min(int(progress), P)
    logical = min(int(progress), P) * (NTREE if progress >= P else 1)

    if progress >= P:
        fx = rx + P * PITCH
        for b in range(NTREE):
            by = rtop - (b + 1) * ROWPITCH
            ax.add_line(Line2D([fx - PITCH / 2, fx + 0.06],
                               [rtop + CELL / 2, by + CELL / 2],
                               color=GREEN, lw=0.8, alpha=0.45, zorder=2))

    for b in range(NTREE):
        y = rtop - (b + 1) * ROWPITCH
        stop = PRUNE_AT.get(b, C)
        lived = int(min(comp, stop))
        pruned_now = b in PRUNE_AT and comp >= PRUNE_AT[b]
        for col in range(lived):
            x = rx + (P + col) * PITCH
            if pruned_now:
                ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor="none",
                                       edgecolor=WASTE, lw=0.8, zorder=3))
            else:
                color = GREEN if (b == 0 and comp >= C) else WHITE
                ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=color, zorder=3))
        tree_computed += lived
        logical += lived + (0 if pruned_now else 0)
        physical += 0 if pruned_now else lived
        if pruned_now and comp < PRUNE_AT[b] + 5:
            ax.text(rx + (P + stop) * PITCH + 0.12, y + 0.01,
                    f"pruned  -{stop} pages", fontsize=8.5, color=DIM,
                    fontproperties=SEG_REG)

    reuse = logical / max(physical, 1)
    ax.text(rx, 2.30, f"tokens computed {tree_computed:4d}", fontsize=11.5,
            color=WHITE, fontproperties=MONO)
    ax.text(rx + 3.1, 2.30, f"KV pages held {physical:4d}", fontsize=11.5,
            color=WHITE, fontproperties=MONO)
    if progress > P:
        ax.text(rx, 1.85, f"KV reuse {reuse:4.2f}x", fontsize=11.5, color=GREEN,
                fontproperties=MONO)
        ax.text(rx + 3.1, 1.85, "pruned pages return to the pool", fontsize=10,
                color=DIM, fontproperties=SEG_REG)


def draw_endcard(alpha: float) -> None:
    ax.add_patch(Rectangle((0, 0), 12.8, 7.2, facecolor=BG, zorder=10))
    ax.text(6.38, 4.15, "Auto", fontsize=56, color=WHITE, alpha=alpha,
            fontproperties=SEG_SEMI, ha="right", zorder=11)
    ax.text(6.38, 4.15, "Tree", fontsize=56, color=GREEN, alpha=alpha,
            fontproperties=SEG_SEMI, ha="left", zorder=11)
    ax.text(6.4, 3.10, "8.79x KV reuse, measured end to end",
            fontsize=18, color=WHITE, alpha=alpha, fontproperties=SEG_REG,
            ha="center", zorder=11)
    ax.text(6.4, 2.66, "H100, Qwen3-8B, MATH-500, beam-8. Full protocol and data in the repo.",
            fontsize=11.5, color=DIM, alpha=alpha, fontproperties=SEG_REG,
            ha="center", zorder=11)
    ax.text(6.4, 1.98, "github.com/RightNow-AI/autotree", fontsize=13.5,
            color=GREEN, alpha=alpha, fontproperties=MONO, ha="center", zorder=11)


total_frames = int(COLS / COLS_PER_FRAME) + HOLD + ENDCARD
anim = FuncAnimation(fig, draw_frame, frames=total_frames, interval=60)
out = r"C:/Users/jaber/RightNow-Full/AutoTree/assets/autotree-mechanism.gif"
anim.save(out, writer=PillowWriter(fps=16))
print("gif written:", out, "frames:", total_frames)
