"""Animated schematic simulation: AutoTree vs sequential best-of-n serving.

Renders assets/autotree-mechanism.gif (infinite loop, no end card). The
animation is a schematic of the execution model, matching the benchmark
protocol (sequential best-of-4 vs a beam-8 tree). Branch scores are
illustrative; the KV accounting is computed by the same rules the engine uses.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

PALETTES = {
    "dark": dict(
        GREEN="#76B900", GREEN_DIM="#4e7a00",
        WHITE="#ececec", WHITE_DIM="#b9b9b9",
        DIM="#7a7a7a", WASTE="#3a3a3a",
        BG="#0a0a0a", PANEL="#111311", BORDER="#242424", TRACK="#1c1c1c",
    ),
    "light": dict(
        GREEN="#76B900", GREEN_DIM="#5a8c00",
        WHITE="#16161d", WHITE_DIM="#4a4a52",
        DIM="#8a8a90", WASTE="#d7d7db",
        BG="#ffffff", PANEL="#f5f6f4", BORDER="#e3e3e6", TRACK="#e9e9ec",
    ),
}
GREEN = GREEN_DIM = WHITE = WHITE_DIM = DIM = WASTE = BG = PANEL = BORDER = TRACK = ""

SEG_SEMI = font_manager.FontProperties(family="Segoe UI", weight="semibold")
SEG_REG = font_manager.FontProperties(family="Segoe UI", weight="regular")
MONO = font_manager.FontProperties(family="Consolas")

P = 16
C = 12
NSEQ = 4
NTREE = 8
CELL = 0.155
GAP = 0.028
PITCH = CELL + GAP
ROWPITCH = CELL + 2.6 * GAP
COLS = P + C
SPEED = 0.4                     # columns per frame
POOL = 160                      # pool gauge capacity, pages
PRUNE_AT = {1: 4, 2: 6, 3: 3, 5: 8, 6: 5, 7: 10}
FINAL_SCORE = {0: 1.84, 1: 0.71, 2: 0.55, 3: 0.32, 4: 1.62, 5: 0.88, 6: 0.44, 7: 1.02}
HOLD = 22

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(PALETTES["dark"]["BG"])
ax = fig.add_axes([0, 0, 1, 1])


def blend(c1: str, c2: str, t: float) -> tuple:
    a = matplotlib.colors.to_rgb(c1)
    b = matplotlib.colors.to_rgb(c2)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def cell(x: float, y: float, face, edge=None, lw: float = 0.0, alpha: float = 1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), CELL, CELL,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=face, edgecolor=edge or face, lw=lw, alpha=alpha, zorder=3))


def token_color(base_new: str, base_old: str, age: float) -> tuple:
    return blend(base_new, base_old, min(1.0, age / 5.0))


def axis_ticks(x0: float, y: float) -> None:
    for tick in range(0, COLS + 1, 8):
        tx = x0 + tick * PITCH
        ax.add_line(Line2D([tx, tx], [y, y + 0.05], color=BORDER, lw=1.0))
        ax.text(tx, y - 0.16, str(tick), fontsize=7.5, color=DIM,
                fontproperties=MONO, ha="center")


def pool_gauge(x0: float, y: float, used: int, color: str, label: str) -> None:
    width = 5.35
    ax.add_patch(Rectangle((x0, y), width, 0.14, facecolor=TRACK, zorder=2))
    frac = min(1.0, used / POOL)
    ax.add_patch(Rectangle((x0, y), width * frac, 0.14, facecolor=color,
                           alpha=0.85, zorder=3))
    ax.text(x0, y + 0.24, label, fontsize=8.5, color=DIM, fontproperties=SEG_REG)
    ax.text(x0 + width, y + 0.24, f"{used}/{POOL} pages", fontsize=8.5,
            color=DIM, fontproperties=MONO, ha="right")


def frontier(x0: float, ytop: float, ybot: float, progress: float, color: str) -> None:
    if 0 < progress < COLS:
        fx = x0 + progress * PITCH
        ax.add_line(Line2D([fx, fx], [ybot, ytop], color=color, lw=1.0,
                           alpha=0.35, zorder=4))


def fork_curve(x1, y1, x2, y2):
    mx = x1 + 0.28
    path = Path([(x1, y1), (mx, y1), (mx, y2), (x2, y2)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=GREEN, lw=0.9,
                           alpha=0.5, zorder=2))


def draw_frame(frame: int) -> None:
    ax.clear()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 12.8, 7.2, facecolor=BG, zorder=0))

    raw = frame * SPEED
    progress = min(raw, COLS)
    comp = max(0.0, progress - P)
    overtime = max(0.0, raw - COLS)

    # header
    ax.text(0.55, 6.80, "One prompt, many candidate answers",
            fontsize=17, color=WHITE, fontproperties=SEG_SEMI)
    ax.text(0.55, 6.44,
            "Schematic simulation of the execution model. Each square is one token of KV cache.",
            fontsize=10.5, color=DIM, fontproperties=SEG_REG)
    ax.text(11.55, 6.80, f"t = {progress:04.1f}", fontsize=12, color=GREEN,
            fontproperties=MONO, ha="right")
    ax.text(11.78, 6.80, "Auto", fontsize=13, color=WHITE, fontproperties=SEG_SEMI)
    ax.text(12.25, 6.80, "Tree", fontsize=13, color=GREEN, fontproperties=SEG_SEMI)

    # ---------------- left panel ----------------
    lx = 0.62
    ax.add_patch(FancyBboxPatch((lx - 0.28, 1.45), 6.0, 4.55,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=PANEL, edgecolor=BORDER, lw=1.0, zorder=1))
    ax.text(lx, 5.66, "STOCK SERVING", fontsize=13, color=WHITE,
            fontproperties=SEG_SEMI)
    ax.text(lx + 2.08, 5.66, "sequential best-of-4", fontsize=11, color=DIM,
            fontproperties=SEG_REG)

    top = 5.22
    seq_tokens = 0
    for row in range(NSEQ):
        y = top - row * ROWPITCH
        for col in range(int(progress)):
            x = lx + col * PITCH
            age = progress - col
            if col < P:
                base = (DIM, WASTE) if row > 0 else (WHITE_DIM, DIM)
                cell(x, y, token_color(base[0], base[1], age))
            else:
                cell(x, y, token_color(WHITE, WHITE_DIM, age))
            seq_tokens += 1
    frontier(lx, top + CELL + 0.05, top - (NSEQ - 1) * ROWPITCH - 0.05,
             progress, WHITE)
    axis_ticks(lx, top - NSEQ * ROWPITCH - 0.06)
    if progress > P:
        ax.text(lx + P * PITCH / 2, top - NSEQ * ROWPITCH - 0.52,
                "prompt recomputed and stored 4x", fontsize=9.5, color=DIM,
                fontproperties=SEG_REG, ha="center")

    pool_gauge(lx, 2.32, seq_tokens, WHITE_DIM, "KV pool")
    ax.text(lx, 1.95, f"tokens computed {seq_tokens:4d}", fontsize=11,
            color=WHITE, fontproperties=MONO)
    ax.text(lx + 3.15, 1.95, "no sharing, no pruning", fontsize=9.5, color=DIM,
            fontproperties=SEG_REG)

    # ---------------- right panel ----------------
    rx = 6.95
    ax.add_patch(FancyBboxPatch((rx - 0.28, 1.45), 6.0, 4.55,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=PANEL, edgecolor=BORDER, lw=1.0, zorder=1))
    ax.text(rx, 5.66, "AUTOTREE", fontsize=13, color=GREEN, fontproperties=SEG_SEMI)
    ax.text(rx + 1.58, 5.66, "beam-8 tree, one shared prefix", fontsize=11,
            color=DIM, fontproperties=SEG_REG)

    rtop = 5.22
    for col in range(min(int(progress), P)):
        age = progress - col
        cell(rx + col * PITCH, rtop, token_color(GREEN, GREEN_DIM, age))
    tree_computed = min(int(progress), P)
    physical = min(int(progress), P)
    logical = min(int(progress), P) * (NTREE if progress >= P else 1)

    if progress >= P:
        fx = rx + P * PITCH
        for b in range(NTREE):
            by = rtop - (b + 1) * ROWPITCH
            fork_curve(fx - PITCH / 2, rtop + CELL / 2, fx + 0.04, by + CELL / 2)

    for b in range(NTREE):
        y = rtop - (b + 1) * ROWPITCH
        stop = PRUNE_AT.get(b, C)
        lived = int(min(comp, stop))
        is_pruned = b in PRUNE_AT and comp >= PRUNE_AT[b]
        fade = min(1.0, (comp - PRUNE_AT[b]) / 3.0) if is_pruned else 0.0
        winner_done = b == 0 and comp >= C
        for col in range(lived):
            x = rx + (P + col) * PITCH
            if is_pruned:
                if fade < 1.0:
                    cell(x, y, WHITE_DIM, alpha=(1 - fade) * 0.8)
                cell(x, y, "none", edge=WASTE, lw=0.8, alpha=1.0)
            else:
                age = comp - col
                if winner_done:
                    sweep = overtime * 1.6
                    color = token_color(GREEN, GREEN_DIM, 2) if col <= sweep else token_color(WHITE, WHITE_DIM, age)
                else:
                    color = token_color(WHITE, WHITE_DIM, age)
                cell(x, y, color)
        tree_computed += lived
        logical += lived
        physical += 0 if is_pruned else lived

        # per-branch scheduler score
        if comp > 0 and lived > 0:
            grow = min(1.0, (lived / C) + 0.15)
            score = FINAL_SCORE[b] * grow
            sx = rx + (P + C) * PITCH + 0.16
            if is_pruned:
                ax.text(sx, y + 0.01, f"{score:+.2f}", fontsize=8.5, color=WASTE,
                        fontproperties=MONO)
            else:
                col_s = GREEN if (winner_done or (b == 0 and comp >= C - 1)) else DIM
                ax.text(sx, y + 0.01, f"{score:+.2f}", fontsize=8.5, color=col_s,
                        fontproperties=MONO)
        if is_pruned and comp < PRUNE_AT[b] + 4:
            label_x = min(rx + (P + stop) * PITCH + 0.16, rx + 4.55)
            ax.text(label_x, y + 0.01, f"pruned -{stop}p", fontsize=8,
                    color=DIM, fontproperties=SEG_REG)

    frontier(rx, rtop + CELL + 0.05, rtop - NTREE * ROWPITCH - 0.05,
             progress, GREEN)
    axis_ticks(rx, rtop - (NTREE + 1) * ROWPITCH + 0.06)

    reuse = logical / max(physical, 1)
    pool_gauge(rx, 2.32, physical, GREEN, "KV pool")
    ax.text(rx, 1.95, f"tokens computed {tree_computed:4d}", fontsize=11,
            color=WHITE, fontproperties=MONO)
    if progress > P:
        ax.text(rx + 3.15, 1.95, f"KV reuse {reuse:4.2f}x", fontsize=11.5,
                color=GREEN, fontproperties=MONO)

    # footer
    ax.text(12.25, 1.02, "github.com/RightNow-AI/autotree", fontsize=10.5,
            color=GREEN, fontproperties=MONO, ha="right")


total_frames = int(COLS / SPEED) + HOLD
for mode, palette in PALETTES.items():
    globals().update(palette)
    fig.patch.set_facecolor(palette["BG"])
    anim = FuncAnimation(fig, draw_frame, frames=total_frames, interval=62)
    out = rf"C:/Users/jaber/RightNow-Full/AutoTree/assets/mechanism-{mode}.gif"
    anim.save(out, writer=PillowWriter(fps=16))
    print("gif written:", out, "frames:", total_frames)
