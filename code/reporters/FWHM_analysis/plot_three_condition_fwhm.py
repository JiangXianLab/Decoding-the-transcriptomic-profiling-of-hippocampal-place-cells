#!/usr/bin/env python
"""Plot 750-PAGFP beside paired 830-PAGFP/PAmCherry FWHM_Z data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_paired_fwhm import REPORTERS, apply_style


CONDITIONS = (
    "750 nm → PAGFP",
    "830 nm → PAGFP",
    "830 nm → PAmCherry",
)
COLORS = {
    "750 nm → PAGFP": "#8DBDD7",
    "830 nm → PAGFP": "#4F8EB8",
    "830 nm → PAmCherry": "#E69F00",
}


def build_condition_table(data: pd.DataFrame) -> pd.DataFrame:
    """Infer 750-only and paired 830 conditions from reporter availability."""
    required = {"animal_id", "roi_name", "roi_label", "reporter", "fwhm_z_um"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = data.loc[data["reporter"].isin(REPORTERS)].copy()
    data["fwhm_z_um"] = pd.to_numeric(data["fwhm_z_um"], errors="raise")
    reporter_count = data.groupby("roi_label")["reporter"].nunique()
    paired_labels = set(reporter_count.loc[reporter_count == 2].index)

    conditions = []
    for row in data.itertuples(index=False):
        is_paired = row.roi_label in paired_labels
        if row.reporter == "PAGFP" and not is_paired:
            condition = CONDITIONS[0]
            activation_nm = 750
            paired_830 = False
        elif row.reporter == "PAGFP" and is_paired:
            condition = CONDITIONS[1]
            activation_nm = 830
            paired_830 = True
        elif row.reporter == "PAmCherry" and is_paired:
            condition = CONDITIONS[2]
            activation_nm = 830
            paired_830 = True
        else:
            raise ValueError(
                f"PAmCherry ROI without matched PAGFP cannot be assigned: {row.roi_label}"
            )
        conditions.append(
            {
                "animal_id": row.animal_id,
                "roi_name": row.roi_name,
                "roi_label": row.roi_label,
                "activation_nm": activation_nm,
                "reporter": row.reporter,
                "condition": condition,
                "paired_830": paired_830,
                "fwhm_z_um": row.fwhm_z_um,
                "grouping_basis": (
                    "matched PAGFP/PAmCherry ROI label"
                    if is_paired
                    else "PAGFP ROI without matched PAmCherry label"
                ),
            }
        )
    result = pd.DataFrame(conditions)
    result["condition_order"] = result["condition"].map(
        {name: index for index, name in enumerate(CONDITIONS)}
    )
    result["roi_number"] = (
        result["roi_name"].str.extract(r"(\d+)", expand=False).astype(int)
    )
    return result.sort_values(
        ["condition_order", "animal_id", "roi_number"]
    ).drop(columns=["condition_order", "roi_number"])


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        values = data.loc[data["condition"] == condition, "fwhm_z_um"]
        rows.append(
            {
                "condition": condition,
                "n_roi": len(values),
                "mean_fwhm_z_um": values.mean(),
                "sd_fwhm_z_um": values.std(ddof=1),
                "sem_fwhm_z_um": values.sem(ddof=1),
                "median_fwhm_z_um": values.median(),
            }
        )
    return pd.DataFrame(rows)


def _symmetric_offsets(count: int, width: float = 0.13) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def plot_conditions(data: pd.DataFrame):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(1.625, 1.5), constrained_layout=True)
    x_positions = {condition: index for index, condition in enumerate(CONDITIONS)}

    # Draw paired 830 nm trajectories using the same horizontal offset at both ends.
    paired = data.loc[data["paired_830"]].pivot(
        index=["animal_id", "roi_name", "roi_label"],
        columns="reporter",
        values="fwhm_z_um",
    ).reset_index()
    paired = paired.sort_values(["animal_id", "roi_name"]).reset_index(drop=True)
    paired_offsets = _symmetric_offsets(len(paired), width=0.10)
    for (_, row), offset in zip(paired.iterrows(), paired_offsets):
        ax.plot(
            [1 + offset, 2 + offset],
            [row["PAGFP"], row["PAmCherry"]],
            color="#B7BCC2",
            alpha=0.48,
            linewidth=0.6,
            zorder=1,
        )
        for x, reporter, condition in (
            (1 + offset, "PAGFP", CONDITIONS[1]),
            (2 + offset, "PAmCherry", CONDITIONS[2]),
        ):
            ax.scatter(
                x,
                row[reporter],
                s=25,
                marker="o",
                facecolor="white",
                edgecolor=COLORS[condition],
                linewidth=1.05,
                zorder=3,
            )

    # The 750 nm PAGFP group is independent of the paired 830 nm ROI set.
    group_750 = data.loc[data["condition"] == CONDITIONS[0]].copy()
    group_750 = group_750.sort_values(["animal_id", "roi_name"])
    offsets = _symmetric_offsets(len(group_750), width=0.13)
    ax.scatter(
        offsets,
        group_750["fwhm_z_um"],
        s=25,
        marker="o",
        facecolor="white",
        edgecolor=COLORS[CONDITIONS[0]],
        linewidth=1.05,
        zorder=3,
    )

    for condition in CONDITIONS:
        x = x_positions[condition]
        values = data.loc[data["condition"] == condition, "fwhm_z_um"].to_numpy()
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        color = COLORS[condition]
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3.0,
            capthick=1.4,
            zorder=4,
        )
        ax.plot(
            [x - 0.13, x + 0.13],
            [mean, mean],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=5,
        )
        ax.text(
            x,
            29.2,
            f"n={len(values)}",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#4D5359",
        )

    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(0, 30)
    ax.set_xticks(
        [0, 1, 2],
        ["PAGFP\n750 nm", "PAGFP\n830 nm", "PAmCherry\n830 nm"],
    )
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_ylabel(r"FWHM$_Z$ (µm)")

    return fig


def run(input_csv: Path, output_dir: Path, prefix: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_condition_table(pd.read_csv(input_csv))
    summary = summarize(data)
    paths = {
        "source_csv": output_dir / f"{prefix}_source_data.csv",
        "summary_csv": output_dir / f"{prefix}_summary.csv",
        "png": output_dir / f"{prefix}.png",
        "pdf": output_dir / f"{prefix}.pdf",
        "svg": output_dir / f"{prefix}.svg",
    }
    data.to_csv(paths["source_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    fig = plot_conditions(data)
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
    parser.add_argument("--prefix", default="A76_A82_750_830_three_condition_fwhm")
    args = parser.parse_args()
    paths = run(args.input_csv, args.output_dir, args.prefix)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
