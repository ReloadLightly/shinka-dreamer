"""Chromatic Field v1: local rendering tokens for scientific project visuals.

Keep visual encodings here; statistical decisions belong to the evidence code.
See docs/VISUAL_STYLE.md. This module never loads data or runs an experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

STYLE_VERSION = "Chromatic Field v1"
BACKGROUND = "#FFFEFC"
TEXT = "#161625"
SECONDARY = "#6F6A78"
COBALT = "#3534CF"
MAGENTA = "#C93683"
ORANGE = "#F26A37"
RULE = "#DCD7E0"
MUTED = "#F1EEF3"
FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
FONT_SMALL = 10
FONT_BASE = 11
FONT_TITLE = 16
FONT_PANEL = 12
FONT_AXIS = 11
FONT_TICK = 10
FONT_MONO = 10
FIGURE_WIDTH = 10.0
DPI = 180
PAD = 0.14
FIELD_STOPS = ("#FCB394", "#FADACD", "#F7F1F3", "#B8B3EE", "#655CDD", "#3534CF")
COLORS = {
    "backprop-neat": COBALT,
    "homogeneous-tanh": MAGENTA,
    "evolution-only": ORANGE,
    "random-search": "#8670AC",
    "fixed-mlp": SECONDARY,
    "logistic": "#A78066",
}
MARKERS = dict(zip(COLORS, ("o", "s", "^", "D", "v", "P"), strict=True))
LINESTYLES = dict(zip(
    COLORS, ("-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 2))), strict=True,
))
METHOD_LABELS = {
    "backprop-neat": "Heterogeneous",
    "homogeneous-tanh": "Tanh only",
    "evolution-only": "Evolution only",
    "random-search": "Random search",
    "fixed-mlp": "Fixed MLP",
    "logistic": "Logistic",
}


def field_cmap() -> LinearSegmentedColormap:
    """P(class 1): peach at zero, cobalt at one; never a method identity scale."""
    return LinearSegmentedColormap.from_list("chromatic-field-v1", FIELD_STOPS, N=256)


def apply_theme() -> None:
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": FONT_BASE,
        "font.monospace": [MONO],
        "axes.titlesize": FONT_PANEL,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": FONT_AXIS,
        "axes.labelpad": 7,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_SMALL,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "axes.edgecolor": RULE,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "grid.color": RULE,
        "grid.linewidth": 0.65,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": BACKGROUND,
        "savefig.transparent": False,
        "savefig.dpi": DPI,
        "figure.dpi": 96,
        "svg.fonttype": "none",
        "svg.hashsalt": "actir-chromatic-field-v1",
        "pdf.fonttype": 42,
    })


def save_figure(fig, path: Path, formats=("svg", "pdf", "png")) -> None:
    """Atomic exports with explicit opaque backgrounds and deterministic metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_facecolor(BACKGROUND)
    fig.patch.set_alpha(1)
    for ax in fig.axes:
        ax.set_facecolor(BACKGROUND)
        ax.patch.set_alpha(1)
    for extension in formats:
        metadata = {"Creator": STYLE_VERSION}
        if extension == "svg":
            metadata["Date"] = None
        elif extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        destination = path.with_suffix(f".{extension}")
        temporary = destination.with_name(f".{destination.stem}.tmp.{extension}")
        fig.savefig(
            temporary, dpi=DPI, bbox_inches="tight", pad_inches=PAD,
            transparent=False, facecolor=BACKGROUND, edgecolor=BACKGROUND, metadata=metadata,
        )
        if extension == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        temporary.replace(destination)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
        fontfamily=MONO, fontsize=FONT_MONO, color=SECONDARY,
    )


def method_handles(methods=None, *, lines=False) -> list[Line2D]:
    return [
        Line2D(
            [], [], color=COLORS[method], marker=MARKERS[method], markersize=6,
            linestyle=LINESTYLES[method] if lines else "", label=METHOD_LABELS[method],
        )
        for method in (methods or COLORS)
    ]
