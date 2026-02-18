from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def save_scatter_heatmap(
    *,
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    out_png: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_log: bool = False,
    y_log: bool = False,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    marker: str = "s",
) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(x, y, c=z, cmap=cmap, s=160, marker=marker, edgecolors="none")
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(sc, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True


def save_line_plot(
    *,
    x: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float]]],
    out_png: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_log: bool = False,
) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, ys in series:
        ax.plot(x, ys, marker="o", linewidth=1.8, label=name)
    if x_log:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True

