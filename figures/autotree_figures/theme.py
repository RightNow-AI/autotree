"""Shared publication styling and deterministic figure writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(
    figure: Figure,
    output_stem: Path,
    *,
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    """Write a figure as 300 dpi PDF and SVG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, dpi=300, bbox_inches="tight")
    return pdf_path, svg_path
