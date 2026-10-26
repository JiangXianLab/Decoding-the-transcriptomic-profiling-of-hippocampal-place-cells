"""Render a single FlowJo-like FACS panel from an exported P3 FCS file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flowkit as fk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        parent
        for parent in (CODE_DIR, *CODE_DIR.parents)
        if (parent / "Data").is_dir() and (parent / "figures").is_dir()
    ),
    Path.cwd(),
)

X_CHANNEL = "FL9-A"
Y_CHANNEL = "FL6-A"
TRANSFORM = {
    "negative": 0.0,
    "width": -100.0,
    "positive": 5.22,
    "max_value": 1677721.6,
}
DEFAULT_POINT_SIZE = 1.8
DEFAULT_TITLE = "750 + 830 nm"
DEFAULT_RASTER_DPI = 600
DISPLAY_MIN = 0.0
DISPLAY_MAX = 4096.0
# The supplied FlowJo panel places the gates at approximately 10^3 (x) and
# 10^4 (y). These are their display-coordinate positions after the exact
# FlowJo-compatible Biex transform above.
DEFAULT_GATE_X = 1350.5902324677672
DEFAULT_GATE_Y = 2314.7213714368745
DEFAULT_BINS = 220
AXIS_TICK_RAW_VALUES = np.array([-1e3, 0.0, 1e3, 1e4, 1e5, 1e6])
AXIS_TICK_LABELS = ["−10³", "0", "10³", "10⁴", "10⁵", "10⁶"]
AXIS_TICK_DISPLAY_VALUES = fk.transforms.WSPBiexTransform(**TRANSFORM).apply(
    AXIS_TICK_RAW_VALUES
)

DENSITY_COLORS = [
    "#22456F",
    "#528FAC",
    "#AECAD8",
    "#F2F2F2",
    "#F7D29D",
    "#EEB84B",
    "#E87A2D",
]
HIGH_CONTRAST_DENSITY_COLORS = [
    "#163A63",
    "#3B729A",
    "#8DB4C6",
    "#E8EEF0",
    "#F4C27A",
    "#DF8A28",
    "#B84A1B",
]
DENSITY_CMAPS = {
    "project": LinearSegmentedColormap.from_list(
        "project_blue_orange", DENSITY_COLORS, N=256
    ),
    "high-contrast": LinearSegmentedColormap.from_list(
        "high_contrast_blue_orange", HIGH_CONTRAST_DENSITY_COLORS, N=256
    ),
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def load_transformed_events(fcs_path: Path, expected_events: int | None = None):
    """Load all events and apply the supplied FlowJo-style Biex transform."""

    fcs_path = Path(fcs_path)
    if not fcs_path.exists():
        raise FileNotFoundError(f"FCS file not found: {fcs_path}")

    sample = fk.Sample(str(fcs_path), use_flowjo_labels=True)
    labels = set(sample.pnn_labels)
    missing = {X_CHANNEL, Y_CHANNEL} - labels
    if missing:
        raise ValueError(f"Missing requested FCS channels: {sorted(missing)}")

    xform = fk.transforms.WSPBiexTransform(**TRANSFORM)
    sample.apply_transform({X_CHANNEL: xform, Y_CHANNEL: xform})
    x = sample.get_channel_events(X_CHANNEL, source="xform", subsample=False).copy()
    y = sample.get_channel_events(Y_CHANNEL, source="xform", subsample=False).copy()

    if expected_events is not None and (x.size != expected_events or y.size != expected_events):
        raise ValueError(
            f"Expected {expected_events} events, got x={x.size}, y={y.size}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Transformed events contain non-finite values")

    metadata = {
        "event_count": int(x.size),
        "x_channel": X_CHANNEL,
        "y_channel": Y_CHANNEL,
        "transform": dict(TRANSFORM),
    }
    return x, y, metadata


def estimate_density(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = DEFAULT_BINS,
    *,
    display_min: float = DISPLAY_MIN,
    display_max: float = DISPLAY_MAX,
):
    """Estimate local density for every event using a two-dimensional histogram."""

    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays with equal shape")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    if display_max <= display_min:
        raise ValueError("display_max must be greater than display_min")
    x_edges = np.linspace(display_min, display_max, bins + 1)
    y_edges = np.linspace(display_min, display_max, bins + 1)
    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    return hist[xi, yi]


def build_figure(
    x: np.ndarray,
    y: np.ndarray,
    *,
    point_size: float = DEFAULT_POINT_SIZE,
    gate_x: float = DEFAULT_GATE_X,
    gate_y: float = DEFAULT_GATE_Y,
    bins: int = DEFAULT_BINS,
    palette: str = "project",
    show_ticks: bool = False,
    title: str = DEFAULT_TITLE,
    rasterized_points: bool = False,
    rasterized_q4: bool = False,
):
    """Build a single panel with optionally rasterized scatter markers."""

    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if not DISPLAY_MIN < gate_x < DISPLAY_MAX or not DISPLAY_MIN < gate_y < DISPLAY_MAX:
        raise ValueError(
            f"gate_x and gate_y must be between {DISPLAY_MIN} and {DISPLAY_MAX}"
        )
    if palette not in DENSITY_CMAPS:
        raise ValueError(f"palette must be one of {sorted(DENSITY_CMAPS)}")
    if rasterized_points and rasterized_q4:
        raise ValueError("rasterized_points and rasterized_q4 are mutually exclusive")

    density = estimate_density(x, y, bins=bins)
    order = np.argsort(density, kind="stable")
    vmax = max(2.0, float(density.max()))

    fig, ax = plt.subplots(figsize=(2.35, 2.35))
    if rasterized_q4:
        q4_mask = (x < gate_x) & (y < gate_y)
        point_layers = ((q4_mask, True), (~q4_mask, False))
    else:
        point_layers = ((np.ones(x.shape, dtype=bool), rasterized_points),)

    for point_mask, rasterized in point_layers:
        layer_order = order[point_mask[order]]
        if layer_order.size == 0:
            continue
        ax.scatter(
            x[layer_order],
            y[layer_order],
            c=density[layer_order],
            cmap=DENSITY_CMAPS[palette],
            norm=LogNorm(vmin=1, vmax=vmax),
            s=point_size,
            linewidths=0,
            edgecolors="none",
            rasterized=rasterized,
        )
    ax.axvline(gate_x, color="black", lw=0.8)
    ax.axhline(gate_y, color="black", lw=0.8)
    ax.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
    ax.set_ylim(DISPLAY_MIN, DISPLAY_MAX)
    if show_ticks:
        ax.set_xticks(AXIS_TICK_DISPLAY_VALUES)
        ax.set_yticks(AXIS_TICK_DISPLAY_VALUES)
        ax.set_xticklabels(AXIS_TICK_LABELS)
        ax.set_yticklabels(AXIS_TICK_LABELS)
        ax.tick_params(axis="both", which="major", length=3, width=0.6, pad=2, labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=4)
    ax.set_box_aspect(1)
    fig.tight_layout(pad=0.5)
    return fig


def save_outputs(
    fig,
    output_stem: Path,
    metadata: dict,
    formats=("svg", "pdf", "png"),
    raster_dpi: int = DEFAULT_RASTER_DPI,
):
    """Save selected figure formats and the accompanying metadata."""

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if raster_dpi <= 0:
        raise ValueError("raster_dpi must be positive")
    paths = []
    format_kwargs = {
        "svg": {"dpi": raster_dpi},
        "pdf": {"dpi": raster_dpi},
        "png": {"dpi": 600},
    }
    unsupported = set(formats) - set(format_kwargs)
    if unsupported:
        raise ValueError(f"Unsupported output formats: {sorted(unsupported)}")
    for suffix in formats:
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **format_kwargs[suffix])
        paths.append(path)

    metadata_path = output_stem.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths.append(metadata_path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures/PlaceCells/01_MainFigures/Figure1/03_Data/FACS/FACS_python",
    )
    parser.add_argument("--output-name", default="PC118_750_830_P3")
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--raster-dpi", type=int, default=DEFAULT_RASTER_DPI)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--gate-x", type=float, default=DEFAULT_GATE_X)
    parser.add_argument("--gate-y", type=float, default=DEFAULT_GATE_Y)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--palette",
        choices=sorted(DENSITY_CMAPS),
        default="project",
        help="density color palette",
    )
    parser.add_argument(
        "--show-ticks",
        action="store_true",
        help="show simplified FlowJo-style Biex tick labels",
    )
    raster_group = parser.add_mutually_exclusive_group()
    raster_group.add_argument(
        "--rasterized-points",
        action="store_true",
        help="rasterize the point layer inside SVG/PDF for faster opening",
    )
    raster_group.add_argument(
        "--rasterized-q4",
        action="store_true",
        help="rasterize only the lower-left Q4 point layer",
    )
    parser.add_argument(
        "--svg-only",
        action="store_true",
        help="write only SVG plus metadata",
    )
    args = parser.parse_args(argv)

    x, y, metadata = load_transformed_events(args.input)
    metadata.update(
        {
            "point_size": args.point_size,
            "gate_x": args.gate_x,
            "gate_y": args.gate_y,
            "density_bins": args.bins,
            "density_palette": (
                DENSITY_COLORS
                if args.palette == "project"
                else HIGH_CONTRAST_DENSITY_COLORS
            ),
            "palette_name": args.palette,
            "show_ticks": args.show_ticks,
            "title": args.title,
            "rasterized_points": args.rasterized_points,
            "rasterized_q4": args.rasterized_q4,
            "raster_dpi": args.raster_dpi,
        }
    )
    fig = build_figure(
        x,
        y,
        point_size=args.point_size,
        gate_x=args.gate_x,
        gate_y=args.gate_y,
        bins=args.bins,
        palette=args.palette,
        show_ticks=args.show_ticks,
        title=args.title,
        rasterized_points=args.rasterized_points,
        rasterized_q4=args.rasterized_q4,
    )
    output_stem = args.output_dir / args.output_name
    formats = ("svg",) if args.svg_only else ("svg", "pdf", "png")
    paths = save_outputs(
        fig,
        output_stem,
        metadata,
        formats=formats,
        raster_dpi=args.raster_dpi,
    )
    plt.close(fig)
    print(f"events={metadata['event_count']}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
