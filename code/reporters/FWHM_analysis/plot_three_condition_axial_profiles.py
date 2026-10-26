#!/usr/bin/env python
"""Plot peak-aligned axial profiles for 750/830 nm reporter conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_peak_aligned_profiles import apply_publication_style


CONDITIONS = (
    "750 nm -> PAGFP",
    "830 nm -> PAGFP",
    "830 nm -> PAmCherry",
)
COLORS = {
    "750 nm -> PAGFP": "#8DBDD7",
    "830 nm -> PAGFP": "#4F8EB8",
    "830 nm -> PAmCherry": "#E69F00",
}


def assign_conditions(aligned: pd.DataFrame) -> pd.DataFrame:
    """Assign aligned ROI profiles to 750-only or paired 830 conditions."""
    required = {
        "group",
        "axial_distance_from_peak_um",
        "roi_name",
        "normalized_fluorescence",
    }
    missing = required.difference(aligned.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = aligned.copy()
    data["normalized_fluorescence"] = pd.to_numeric(
        data["normalized_fluorescence"], errors="coerce"
    )
    groups_by_roi = data.groupby("roi_name")["group"].agg(set).to_dict()
    rows = []
    for row in data.itertuples(index=False):
        roi_groups = groups_by_roi[row.roi_name]
        is_paired = {"PAGFP", "PAmCherry"}.issubset(roi_groups)
        if row.group == "PAGFP" and not is_paired:
            condition = CONDITIONS[0]
            activation_nm = 750
        elif row.group == "PAGFP" and is_paired:
            condition = CONDITIONS[1]
            activation_nm = 830
        elif row.group == "PAmCherry" and is_paired:
            condition = CONDITIONS[2]
            activation_nm = 830
        else:
            raise ValueError(
                f"PAmCherry ROI without matched PAGFP cannot be assigned: {row.roi_name}"
            )
        animal_id, roi_label = str(row.roi_name).split(":", 1)
        rows.append(
            {
                "animal_id": animal_id,
                "roi_name": roi_label,
                "roi_key": row.roi_name,
                "activation_nm": activation_nm,
                "reporter": row.group,
                "condition": condition,
                "axial_distance_from_peak_um": float(
                    row.axial_distance_from_peak_um
                ),
                "normalized_fluorescence": row.normalized_fluorescence,
                "grouping_basis": (
                    "matched PAGFP/PAmCherry ROI label"
                    if is_paired
                    else "PAGFP ROI without matched PAmCherry label"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["condition_order"] = result["condition"].map(
        {name: index for index, name in enumerate(CONDITIONS)}
    )
    return result.sort_values(
        ["condition_order", "animal_id", "roi_name", "axial_distance_from_peak_um"]
    ).drop(columns="condition_order")


def summarize_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pointwise mean, SD, SEM, and ROI count for each condition."""
    frames = []
    for condition in CONDITIONS:
        frame = data.loc[data["condition"] == condition]
        summary = (
            frame.groupby("axial_distance_from_peak_um", sort=True)[
                "normalized_fluorescence"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_normalized_fluorescence",
                    "std": "sd",
                    "count": "n_roi",
                }
            )
        )
        summary.insert(0, "condition", condition)
        summary["sem"] = summary["sd"] / np.sqrt(summary["n_roi"])
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def plot_profiles(summary: pd.DataFrame):
    """Draw three peak-aligned mean profiles with pointwise SEM ribbons."""
    import matplotlib.pyplot as plt

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)
    legend_labels = {
        CONDITIONS[0]: "PAGFP, 750 nm",
        CONDITIONS[1]: "PAGFP, 830 nm",
        CONDITIONS[2]: "PAmCherry, 830 nm",
    }
    for condition in CONDITIONS:
        frame = summary.loc[summary["condition"] == condition].sort_values(
            "axial_distance_from_peak_um"
        )
        x = frame["axial_distance_from_peak_um"].to_numpy(dtype=float)
        mean = frame["mean_normalized_fluorescence"].to_numpy(dtype=float)
        sem = frame["sem"].to_numpy(dtype=float)
        color = COLORS[condition]
        n_roi = int(frame["n_roi"].max())
        label = f"{legend_labels[condition]} (n={n_roi})"
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.22, lw=0)
        ax.plot(x, mean, color=color, linewidth=1.55, label=label)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Axial distance from peak (µm)")
    ax.set_ylabel("Normalized fluorescence")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.8,
        handlelength=1.45,
        labelspacing=0.28,
        borderaxespad=0.25,
    )
    return fig


def run(input_csv: Path, output_dir: Path, prefix: str) -> dict[str, object]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assigned = assign_conditions(pd.read_csv(input_csv))
    summary = summarize_profiles(assigned)
    paths = {
        "source_csv": output_dir / f"{prefix}_source_data.csv",
        "summary_csv": output_dir / f"{prefix}_summary.csv",
        "png": output_dir / f"{prefix}.png",
        "pdf": output_dir / f"{prefix}.pdf",
        "svg": output_dir / f"{prefix}.svg",
    }
    assigned.to_csv(paths["source_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    fig = plot_profiles(summary)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"assigned": assigned, "summary": summary, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix", default="A76_A82_750_830_three_condition_axial_profile"
    )
    args = parser.parse_args()
    result = run(args.input_csv, args.output_dir, args.prefix)
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
