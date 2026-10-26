#!/usr/bin/env python
"""Create a publication-style paired FWHM_Z reporter comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPORTERS = ("PAGFP", "PAmCherry")
COLORS = {"PAGFP": "#6B9FC6", "PAmCherry": "#E69F00"}
MARKERS = {"A76": "o", "A82": "s"}


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 3.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def build_paired_table(data: pd.DataFrame) -> pd.DataFrame:
    """Keep ROI identities with measurements for both reporters."""
    required = {"animal_id", "roi_name", "roi_label", "reporter", "fwhm_z_um"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    subset = data.loc[data["reporter"].isin(REPORTERS)].copy()
    subset["fwhm_z_um"] = pd.to_numeric(subset["fwhm_z_um"], errors="raise")
    counts = subset.groupby("roi_label")["reporter"].nunique()
    paired_labels = counts.loc[counts == len(REPORTERS)].index
    paired = subset.loc[subset["roi_label"].isin(paired_labels)].copy()
    duplicates = paired.duplicated(["roi_label", "reporter"], keep=False)
    if duplicates.any():
        labels = sorted(paired.loc[duplicates, "roi_label"].unique())
        raise ValueError(f"duplicate reporter values for ROI labels: {labels}")
    order = {name: index for index, name in enumerate(REPORTERS)}
    paired["reporter_order"] = paired["reporter"].map(order)
    paired["roi_number"] = (
        paired["roi_name"].str.extract(r"(\d+)", expand=False).astype(int)
    )
    return paired.sort_values(
        ["animal_id", "roi_number", "reporter_order"]
    ).drop(columns=["reporter_order", "roi_number"])


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reporter in REPORTERS:
        values = paired.loc[paired["reporter"] == reporter, "fwhm_z_um"]
        rows.append(
            {
                "reporter": reporter,
                "n_paired_roi": len(values),
                "mean_fwhm_z_um": values.mean(),
                "sd_fwhm_z_um": values.std(ddof=1),
                "sem_fwhm_z_um": values.sem(ddof=1),
                "median_fwhm_z_um": values.median(),
            }
        )
    return pd.DataFrame(rows)


def plot_paired(paired: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    fig, ax = plt.subplots(figsize=(2.35, 2.75), constrained_layout=True)
    wide = paired.pivot(
        index=["animal_id", "roi_name", "roi_label"],
        columns="reporter",
        values="fwhm_z_um",
    ).reset_index()

    for _, row in wide.iterrows():
        ax.plot(
            [0, 1],
            [row[REPORTERS[0]], row[REPORTERS[1]]],
            color="#A7ADB4",
            alpha=0.55,
            linewidth=0.65,
            zorder=1,
        )
        marker = MARKERS.get(row["animal_id"], "o")
        for x, reporter in enumerate(REPORTERS):
            ax.scatter(
                x,
                row[reporter],
                s=28,
                marker=marker,
                facecolor="white",
                edgecolor=COLORS[reporter],
                linewidth=1.15,
                zorder=3,
            )

    for x, reporter in enumerate(REPORTERS):
        values = wide[reporter].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt="none",
            ecolor=COLORS[reporter],
            elinewidth=1.5,
            capsize=3.5,
            capthick=1.5,
            zorder=4,
        )
        ax.plot(
            [x - 0.13, x + 0.13],
            [mean, mean],
            color=COLORS[reporter],
            linewidth=2.3,
            solid_capstyle="round",
            zorder=5,
        )

    y_max = max(30.0, np.ceil(wide[list(REPORTERS)].to_numpy().max() / 5.0) * 5.0)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0, y_max)
    ax.set_xticks([0, 1], REPORTERS)
    ax.set_yticks(np.arange(0, y_max + 0.1, 5.0))
    ax.set_ylabel(r"FWHM$_Z$ (µm)")
    ax.set_xlabel("")

    animal_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#626A73",
            markeredgewidth=1.0,
            markersize=4.8,
            label=animal,
        )
        for animal, marker in MARKERS.items()
        if animal in set(wide["animal_id"])
    ]
    ax.legend(
        handles=animal_handles,
        loc="upper left",
        frameon=False,
        handletextpad=0.35,
        borderaxespad=0.1,
        labelspacing=0.3,
    )
    return fig


def run(input_csv: Path, output_dir: Path, prefix: str) -> dict[str, Path]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = build_paired_table(pd.read_csv(input_csv))
    summary = summarize(paired)
    paths = {
        "paired_csv": output_dir / f"{prefix}_source_data.csv",
        "summary_csv": output_dir / f"{prefix}_summary.csv",
        "png": output_dir / f"{prefix}.png",
        "pdf": output_dir / f"{prefix}.pdf",
        "svg": output_dir / f"{prefix}.svg",
    }
    paired.to_csv(paths["paired_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    fig = plot_paired(paired)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="A76_A82_830nm_paired_fwhm")
    args = parser.parse_args()
    paths = run(args.input_csv, args.output_dir, args.prefix)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
