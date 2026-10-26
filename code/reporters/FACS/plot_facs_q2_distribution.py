"""Plot Cy3 distributions for Q2 events from the current FACS samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        parent
        for parent in (CODE_DIR, *CODE_DIR.parents)
        if (parent / "Data").is_dir() and (parent / "figures").is_dir()
    ),
    Path.cwd(),
)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from plot_facs_panel import (
    AXIS_TICK_DISPLAY_VALUES,
    DEFAULT_GATE_X,
    DEFAULT_GATE_Y,
    TRANSFORM,
    X_CHANNEL,
    Y_CHANNEL,
    load_transformed_events,
)


FLOW_TICK_LABELS = ["0", "10³", "10⁴", "10⁵", "10⁶"]
Q2_DISPLAY_MIN = float(AXIS_TICK_DISPLAY_VALUES[1])
Q2_DISPLAY_MAX = float(AXIS_TICK_DISPLAY_VALUES[-1])
DEFAULT_INPUTS = [
    ("Control", PROJECT_ROOT / "Data/FACS/export_PC102_NoStiControl_concat_1_P3.fcs"),
    ("750 nm", PROJECT_ROOT / "Data/FACS/export_PC132_750_concat_1_P3.fcs"),
    ("830 nm", PROJECT_ROOT / "Data/FACS/export_2_PC97_830Sti_concat_1_P3.fcs"),
    ("750 & 830 nm", PROJECT_ROOT / "Data/FACS/export_PC118_750_830_concat_1_P3.fcs"),
]


def select_q2_x(
    x: np.ndarray,
    y: np.ndarray,
    gate_x: float = DEFAULT_GATE_X,
    gate_y: float = DEFAULT_GATE_Y,
) -> np.ndarray:
    """Return transformed x-channel values inside the inclusive Q2 gate."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be one-dimensional arrays with equal shape")
    return x[(x >= gate_x) & (y >= gate_y)]


def smoothed_histogram(
    values: np.ndarray,
    edges: np.ndarray,
    sigma_bins: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and a peak-normalized Gaussian-smoothed histogram."""

    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be a strictly increasing one-dimensional array")

    counts, _ = np.histogram(values, bins=edges)
    radius = max(1, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    full = np.convolve(counts.astype(float), kernel, mode="full")
    start = (kernel.size - 1) // 2
    density = full[start : start + counts.size]
    if density.max() > 0:
        density /= density.max()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, density


def build_q2_distribution_figure(
    groups: list[dict],
    bins: int = 64,
    sigma_bins: float = 1.2,
) -> mpl.figure.Figure:
    """Build the final Q2 Cy3 ridgeline panel used beside the FACS plots."""

    if len(groups) != 4:
        raise ValueError("groups must contain exactly four conditions")
    if bins < 8:
        raise ValueError("bins must be at least 8")

    edges = np.linspace(Q2_DISPLAY_MIN, Q2_DISPLAY_MAX, bins + 1)
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    ):
        fig, ax = plt.subplots(figsize=(2.475, 2.55))
        baselines = np.arange(3, -1, -1, dtype=float)
        ridge_height = 0.72

        for baseline, group in zip(baselines, groups, strict=True):
            label = str(group["label"])
            values = np.asarray(group["values"], dtype=float)
            centers, density = smoothed_histogram(
                values, edges, sigma_bins=sigma_bins
            )

            ax.hlines(
                baseline,
                Q2_DISPLAY_MIN,
                Q2_DISPLAY_MAX,
                color="#111111",
                lw=0.8,
                zorder=1,
            )
            if values.size:
                ax.fill_between(
                    centers,
                    baseline,
                    baseline + ridge_height * density,
                    color="#AFAFAF",
                    alpha=1.0,
                    linewidth=0,
                    zorder=2,
                )

            display_label = "750 &\n830 nm" if label == "750 & 830 nm" else label
            ax.text(
                1.035,
                baseline + 0.18,
                display_label,
                ha="left",
                va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=7,
            )

        dim_left = float(AXIS_TICK_DISPLAY_VALUES[2])
        shared_boundary = float(AXIS_TICK_DISPLAY_VALUES[3])
        bright_right = float(AXIS_TICK_DISPLAY_VALUES[4])
        for position, color in (
            (dim_left, "#6EA6C8"),
            (shared_boundary, "#777777"),
            (bright_right, "#E69F00"),
        ):
            ax.plot(
                [position, position],
                [0, 3.42],
                color=color,
                lw=0.8,
                ls=(0, (3, 2)),
                zorder=4,
            )

        ax.text(
            (dim_left + shared_boundary) / 2,
            3.50,
            "dim",
            color="#5B92B7",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        ax.text(
            (shared_boundary + bright_right) / 2,
            3.50,
            "bright",
            color="#D88A00",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        ax.text(
            -0.08,
            1.02,
            "i",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlim(Q2_DISPLAY_MIN, Q2_DISPLAY_MAX)
        ax.set_ylim(-0.08, 3.78)
        ax.set_xticks(AXIS_TICK_DISPLAY_VALUES[1:])
        ax.set_xticklabels(FLOW_TICK_LABELS, fontsize=6.5)
        ax.set_xlabel("PAmCherry (Cy3)", color="#D66A24", labelpad=3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="x", which="major", length=2.5, width=0.7, pad=2)
        fig.subplots_adjust(left=0.08, right=0.76, top=0.96, bottom=0.14)
    return fig


def load_default_groups() -> tuple[list[dict], dict]:
    """Load the four current FCS samples and select Q2 events consistently."""

    groups = []
    files = []
    for label, path in DEFAULT_INPUTS:
        x, y, sample_metadata = load_transformed_events(path)
        values = select_q2_x(x, y)
        groups.append({"label": label, "values": values})
        files.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "total_events": sample_metadata["event_count"],
                "q2_events": int(values.size),
            }
        )

    metadata = {
        "channels": {"x": X_CHANNEL, "y": Y_CHANNEL},
        "transform": dict(TRANSFORM),
        "q2_gate": {"x_min": DEFAULT_GATE_X, "y_min": DEFAULT_GATE_Y},
        "files": files,
    }
    return groups, metadata


def save_outputs(
    fig: mpl.figure.Figure,
    metadata: dict,
    output_stem: Path,
) -> list[Path]:
    """Save editable vector files, a 600-dpi preview, and JSON provenance."""

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for suffix, kwargs in (
            ("svg", {}),
            ("pdf", {}),
            ("png", {"dpi": 600}),
        ):
            path = output_stem.with_suffix(f".{suffix}")
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
            paths.append(path)

    metadata_path = output_stem.parent / f"{output_stem.name}.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.append(metadata_path)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PROJECT_ROOT / "figures/PlaceCells/01_MainFigures/Figure1/03_Data/FACS/FACS_python/q2_distribution/q2_cy3_distribution",
    )
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--sigma-bins", type=float, default=1.2)
    args = parser.parse_args(argv)

    groups, metadata = load_default_groups()
    metadata["plot"] = {
        "type": "peak-normalized smoothed histogram with raw-event rug",
        "bins": args.bins,
        "sigma_bins": args.sigma_bins,
        "display_range": [Q2_DISPLAY_MIN, Q2_DISPLAY_MAX],
    }
    fig = build_q2_distribution_figure(
        groups, bins=args.bins, sigma_bins=args.sigma_bins
    )
    paths = save_outputs(fig, metadata, args.output_stem)
    plt.close(fig)

    for group in groups:
        print(f"{group['label']}: Q2 n={group['values'].size}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
