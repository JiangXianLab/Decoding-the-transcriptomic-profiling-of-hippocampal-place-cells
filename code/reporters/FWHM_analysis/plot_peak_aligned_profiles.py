#!/usr/bin/env python
"""Peak-align and average saved FWHM_Z ROI profiles.

The input directory is expected to contain matching CSV/JSON pairs produced by
``fwhmz_napari_gui.py``. Each ROI is aligned to its own axial fluorescence peak
before group averaging. Exclusions are always explicit; QC flags are reported
but never used to silently remove profiles.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_GROUP_PATTERNS = {
    "PAGFP": "_940_",
    "PAmCherry": "_1070_",
}

DEFAULT_COLORS = {
    "PAGFP": "#6B9FC6",
    "PAmCherry": "#E69F00",
}


@dataclass(frozen=True)
class ProfileRecord:
    group: str
    roi_name: str
    csv_path: Path
    json_path: Path
    profile: pd.DataFrame
    metadata: dict
    peak_z_um: float
    included: bool
    exclusion_reason: str
    qc_flags: tuple[str, ...]
    animal_id: str = ""


def apply_publication_style() -> None:
    """Apply a compact, editable-text matplotlib style."""
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
            "axes.unicode_minus": True,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def infer_group(filename: str, group_patterns: Mapping[str, str]) -> str | None:
    """Return the unique group whose token occurs in ``filename``."""
    matches = [name for name, token in group_patterns.items() if token in filename]
    if len(matches) > 1:
        raise ValueError(f"multiple group patterns matched {filename}: {matches}")
    return matches[0] if matches else None


def _profile_qc_flags(
    profile: pd.DataFrame,
    metadata: Mapping[str, object],
    metric_column: str,
    edge_threshold: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    values = profile[metric_column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        flags.append("non-finite normalized values")
    if values.size < 3:
        flags.append("fewer than 3 Z planes")
    if values.size and int(np.nanargmax(values)) in (0, values.size - 1):
        flags.append("peak at Z-stack boundary")
    if values.size >= 2 and max(abs(values[0]), abs(values[-1])) > edge_threshold:
        flags.append(f"edge intensity exceeds {edge_threshold:.2f}")
    warning = str(metadata.get("warnings", "") or "").strip()
    if warning:
        flags.append(warning)
    return tuple(flags)


def load_profiles(
    input_dir: Path,
    group_patterns: Mapping[str, str] = DEFAULT_GROUP_PATTERNS,
    metric_column: str = "smoothed_normalized_intensity",
    exclusions: Mapping[str, Iterable[str]] | None = None,
    edge_threshold: float = 0.15,
    animal_id: str | None = None,
) -> list[ProfileRecord]:
    """Load matching GUI CSV/JSON outputs and attach explicit inclusion state."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    animal_id = str(animal_id or input_dir.parent.name)

    exclusion_sets = {
        group: {str(name) for name in names}
        for group, names in (exclusions or {}).items()
    }
    records: list[ProfileRecord] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        group = infer_group(csv_path.name, group_patterns)
        if group is None:
            continue
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"missing matching JSON for {csv_path.name}")

        profile = pd.read_csv(csv_path)
        required = {"z_um", metric_column}
        missing = required.difference(profile.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        roi_name = str(metadata.get("roi_name") or csv_path.name.split("_", 1)[0])

        values = profile[metric_column].to_numpy(dtype=float)
        z_um = profile["z_um"].to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            raise ValueError(f"{csv_path.name} has no finite profile values")
        peak_z_um = float(z_um[int(np.nanargmax(values))])

        excluded = roi_name in exclusion_sets.get(group, set())
        reason = "listed in EXCLUDE_ROIS" if excluded else ""
        flags = _profile_qc_flags(profile, metadata, metric_column, edge_threshold)
        records.append(
            ProfileRecord(
                group=group,
                roi_name=roi_name,
                csv_path=csv_path,
                json_path=json_path,
                profile=profile,
                metadata=metadata,
                peak_z_um=peak_z_um,
                included=not excluded,
                exclusion_reason=reason,
                qc_flags=flags,
                animal_id=animal_id,
            )
        )

    if not records:
        raise ValueError(f"no profile CSV/JSON pairs found in {input_dir}")
    missing_groups = set(group_patterns).difference(record.group for record in records)
    if missing_groups:
        raise ValueError(f"no profiles found for groups: {sorted(missing_groups)}")
    return records


def build_qc_table(records: Sequence[ProfileRecord]) -> pd.DataFrame:
    """Create one auditable row per ROI."""
    rows = []
    for record in records:
        profile = record.profile
        values = profile["smoothed_normalized_intensity"].to_numpy(dtype=float)
        rows.append(
            {
                "group": record.group,
                "animal_id": record.animal_id,
                "roi_name": record.roi_name,
                "included": record.included,
                "exclusion_reason": record.exclusion_reason,
                "qc_flags": "; ".join(record.qc_flags),
                "source_csv": record.csv_path.name,
                "peak_z_um": record.peak_z_um,
                "fwhm_z_um": record.metadata.get("fwhm_z_um", np.nan),
                "column_height_20pct_um": record.metadata.get(
                    "column_height_20pct_um", np.nan
                ),
                "gaussian_sigma_planes": record.metadata.get(
                    "gaussian_sigma_planes", np.nan
                ),
                "z_step_um": record.metadata.get("z_step_um", np.nan),
                "edge_left_normalized": values[0],
                "edge_right_normalized": values[-1],
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "roi_name"]).reset_index(drop=True)


def build_common_grid(
    records: Sequence[ProfileRecord],
    plot_range_um: tuple[float, float] = (-50.0, 50.0),
    grid_step_um: float | None = None,
) -> np.ndarray:
    """Build a peak-centered grid without claiming finer sampling than acquired."""
    if grid_step_um is None:
        steps = [
            float(record.metadata.get("z_step_um", np.nan))
            for record in records
            if record.included
        ]
        steps = [step for step in steps if np.isfinite(step) and step > 0]
        if not steps:
            raise ValueError("no positive z_step_um values found")
        grid_step_um = min(steps)
    if grid_step_um <= 0:
        raise ValueError("grid_step_um must be positive")
    start, stop = map(float, plot_range_um)
    if stop <= start:
        raise ValueError("plot_range_um must satisfy stop > start")
    return np.arange(start, stop + grid_step_um * 0.5, grid_step_um)


def align_profiles(
    records: Sequence[ProfileRecord],
    grid_um: np.ndarray,
    metric_column: str = "smoothed_normalized_intensity",
) -> dict[str, pd.DataFrame]:
    """Interpolate included ROI profiles onto a common peak-centered grid."""
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        if not record.included:
            continue
        x = record.profile["z_um"].to_numpy(dtype=float) - record.peak_z_um
        y = record.profile[metric_column].to_numpy(dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        aligned = np.full(grid_um.shape, np.nan, dtype=float)
        inside = (grid_um >= x[0]) & (grid_um <= x[-1])
        aligned[inside] = np.interp(grid_um[inside], x, y)
        key = f"{record.animal_id}:{record.roi_name}"
        if key in matrices.setdefault(record.group, {}):
            raise ValueError(f"duplicate animal/ROI key within {record.group}: {key}")
        matrices[record.group][key] = aligned

    result = {}
    for group, columns in matrices.items():
        result[group] = pd.DataFrame(columns, index=grid_um)
        result[group].index.name = "axial_distance_from_peak_um"
    return result


def summarize_aligned_profiles(
    aligned: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return long-form mean, SD, SEM, and pointwise sample count."""
    frames = []
    for group, matrix in aligned.items():
        count = matrix.count(axis=1)
        sd = matrix.std(axis=1, ddof=1)
        frame = pd.DataFrame(
            {
                "group": group,
                "axial_distance_from_peak_um": matrix.index.to_numpy(dtype=float),
                "mean_normalized_fluorescence": matrix.mean(axis=1),
                "sd": sd,
                "sem": sd / np.sqrt(count.where(count > 0)),
                "n_roi": count,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_roi_metrics(qc_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize saved ROI-level FWHM values without recomputing them."""
    included = qc_table.loc[qc_table["included"]].copy()
    rows = []
    for group, frame in included.groupby("group", sort=False):
        fwhm = pd.to_numeric(frame["fwhm_z_um"], errors="coerce").dropna()
        rows.append(
            {
                "group": group,
                "n_roi": len(frame),
                "fwhm_n": len(fwhm),
                "fwhm_z_um_mean": fwhm.mean(),
                "fwhm_z_um_sd": fwhm.std(ddof=1),
                "fwhm_z_um_sem": fwhm.sem(ddof=1),
                "fwhm_z_um_median": fwhm.median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_roi_metrics_by_animal(qc_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize saved FWHM values separately for each animal and reporter."""
    included = qc_table.loc[qc_table["included"]].copy()
    rows = []
    for (animal_id, group), frame in included.groupby(
        ["animal_id", "group"], sort=False
    ):
        fwhm = pd.to_numeric(frame["fwhm_z_um"], errors="coerce").dropna()
        rows.append(
            {
                "animal_id": animal_id,
                "group": group,
                "n_roi": len(frame),
                "fwhm_n": len(fwhm),
                "fwhm_z_um_mean": fwhm.mean(),
                "fwhm_z_um_sd": fwhm.std(ddof=1),
                "fwhm_z_um_sem": fwhm.sem(ddof=1),
                "fwhm_z_um_median": fwhm.median(),
            }
        )
    return pd.DataFrame(rows)


def plot_mean_profiles(
    summary: pd.DataFrame,
    group_order: Sequence[str] = ("PAGFP", "PAmCherry"),
    colors: Mapping[str, str] = DEFAULT_COLORS,
    band: str = "sem",
    show_n_in_legend: bool = False,
):
    """Plot peak-aligned group means with SD or SEM ribbons."""
    import matplotlib.pyplot as plt

    if band not in {"sem", "sd"}:
        raise ValueError("band must be 'sem' or 'sd'")
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.85), constrained_layout=True)
    for group in group_order:
        frame = summary.loc[summary["group"] == group].sort_values(
            "axial_distance_from_peak_um"
        )
        if frame.empty:
            continue
        x = frame["axial_distance_from_peak_um"].to_numpy(dtype=float)
        mean = frame["mean_normalized_fluorescence"].to_numpy(dtype=float)
        error = frame[band].to_numpy(dtype=float)
        color = colors.get(group, "#444444")
        n_roi = int(frame["n_roi"].max())
        label = f"{group} (n={n_roi} ROIs)" if show_n_in_legend else group
        ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.25, lw=0)
        ax.plot(x, mean, color=color, lw=1.6, label=label)

    x_min = float(summary["axial_distance_from_peak_um"].min())
    x_max = float(summary["axial_distance_from_peak_um"].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.1, 1.1)
    tick_step = 25.0 if (x_max - x_min) >= 80.0 else 10.0
    tick_start = np.ceil(x_min / tick_step) * tick_step
    tick_stop = np.floor(x_max / tick_step) * tick_step
    ax.set_xticks(np.arange(tick_start, tick_stop + tick_step * 0.5, tick_step))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Axial distance from peak (µm)")
    ax.set_ylabel("Normalized fluorescence")
    ax.legend(loc="upper left", bbox_to_anchor=(0.56, 1.02), handlelength=1.5)
    return fig


def plot_individual_profiles(
    aligned: Mapping[str, pd.DataFrame],
    group_order: Sequence[str] = ("PAGFP", "PAmCherry"),
    colors: Mapping[str, str] = DEFAULT_COLORS,
):
    """Create a QC plot showing every aligned ROI and the group mean."""
    import matplotlib.pyplot as plt

    apply_publication_style()
    fig, axes = plt.subplots(1, len(group_order), figsize=(7.0, 2.8), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, group_order):
        matrix = aligned[group]
        color = colors.get(group, "#444444")
        for roi_name in matrix.columns:
            ax.plot(matrix.index, matrix[roi_name], color=color, alpha=0.20, lw=0.7)
        ax.plot(matrix.index, matrix.mean(axis=1), color=color, lw=1.7)
        ax.set_title(f"{group} (n={matrix.shape[1]} ROIs)")
        ax.set_xlabel("Axial distance from peak (µm)")
        ax.set_xlim(matrix.index.min(), matrix.index.max())
        ax.set_ylim(-0.1, 1.1)
    axes[0].set_ylabel("Normalized fluorescence")
    return fig


def save_outputs(
    output_dir: Path,
    prefix: str,
    records: Sequence[ProfileRecord],
    aligned: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    band: str = "sem",
) -> dict[str, Path]:
    """Save source tables plus publication and QC figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_table = build_qc_table(records)
    metric_summary = summarize_roi_metrics(qc_table)

    paths = {
        "qc_csv": output_dir / f"{prefix}_roi_qc.csv",
        "profile_summary_csv": output_dir / f"{prefix}_profile_source_data.csv",
        "metric_summary_csv": output_dir / f"{prefix}_fwhm_summary.csv",
        "animal_metric_summary_csv": output_dir
        / f"{prefix}_fwhm_summary_by_animal.csv",
        "aligned_csv": output_dir / f"{prefix}_aligned_roi_profiles.csv",
        "figure_png": output_dir / f"{prefix}_axial_profile.png",
        "figure_pdf": output_dir / f"{prefix}_axial_profile.pdf",
        "figure_svg": output_dir / f"{prefix}_axial_profile.svg",
        "qc_figure_png": output_dir / f"{prefix}_individual_profiles_qc.png",
    }
    qc_table.to_csv(paths["qc_csv"], index=False)
    summary.to_csv(paths["profile_summary_csv"], index=False)
    metric_summary.to_csv(paths["metric_summary_csv"], index=False)
    summarize_roi_metrics_by_animal(qc_table).to_csv(
        paths["animal_metric_summary_csv"], index=False
    )

    aligned_long = []
    for group, matrix in aligned.items():
        frame = matrix.reset_index().melt(
            id_vars="axial_distance_from_peak_um",
            var_name="roi_name",
            value_name="normalized_fluorescence",
        )
        frame.insert(0, "group", group)
        aligned_long.append(frame)
    pd.concat(aligned_long, ignore_index=True).to_csv(paths["aligned_csv"], index=False)

    fig = plot_mean_profiles(summary, band=band)
    fig.savefig(paths["figure_png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["figure_pdf"], bbox_inches="tight")
    fig.savefig(paths["figure_svg"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    qc_fig = plot_individual_profiles(aligned)
    qc_fig.savefig(paths["qc_figure_png"], dpi=300, bbox_inches="tight")
    plt.close(qc_fig)
    return paths


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    prefix: str = "A76",
    exclusions: Mapping[str, Iterable[str]] | None = None,
    plot_range_um: tuple[float, float] = (-50.0, 50.0),
    grid_step_um: float | None = None,
    band: str = "sem",
) -> dict[str, object]:
    """Run the complete analysis and return in-memory tables and output paths."""
    records = load_profiles(input_dir, exclusions=exclusions)
    grid = build_common_grid(records, plot_range_um, grid_step_um)
    aligned = align_profiles(records, grid)
    summary = summarize_aligned_profiles(aligned)
    paths = save_outputs(output_dir, prefix, records, aligned, summary, band=band)
    return {
        "records": records,
        "qc_table": build_qc_table(records),
        "aligned": aligned,
        "profile_summary": summary,
        "metric_summary": summarize_roi_metrics(build_qc_table(records)),
        "animal_metric_summary": summarize_roi_metrics_by_animal(
            build_qc_table(records)
        ),
        "paths": paths,
    }


def run_multi_animal_analysis(
    input_dirs: Mapping[str, Path],
    output_dir: Path,
    prefix: str = "combined",
    exclusions: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
    plot_range_um: tuple[float, float] = (-50.0, 50.0),
    grid_step_um: float | None = None,
    band: str = "sem",
) -> dict[str, object]:
    """Run one pooled-ROI analysis while retaining each animal identifier."""
    if not input_dirs:
        raise ValueError("input_dirs must contain at least one animal")
    records: list[ProfileRecord] = []
    for animal_id, input_dir in input_dirs.items():
        animal_exclusions = (exclusions or {}).get(animal_id, {})
        records.extend(
            load_profiles(
                input_dir,
                exclusions=animal_exclusions,
                animal_id=animal_id,
            )
        )

    grid = build_common_grid(records, plot_range_um, grid_step_um)
    aligned = align_profiles(records, grid)
    summary = summarize_aligned_profiles(aligned)
    paths = save_outputs(output_dir, prefix, records, aligned, summary, band=band)
    qc_table = build_qc_table(records)
    return {
        "records": records,
        "qc_table": qc_table,
        "aligned": aligned,
        "profile_summary": summary,
        "metric_summary": summarize_roi_metrics(qc_table),
        "animal_metric_summary": summarize_roi_metrics_by_animal(qc_table),
        "paths": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="A76")
    parser.add_argument("--plot-min", type=float, default=-50.0)
    parser.add_argument("--plot-max", type=float, default=50.0)
    parser.add_argument("--grid-step", type=float, default=None)
    parser.add_argument("--band", choices=("sem", "sd"), default="sem")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GROUP:ROI",
        help="Explicitly exclude one ROI, e.g. PAGFP:13A. Repeatable.",
    )
    return parser


def _parse_exclusions(values: Sequence[str]) -> dict[str, set[str]]:
    exclusions: dict[str, set[str]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"invalid exclusion {value!r}; expected GROUP:ROI")
        group, roi_name = value.split(":", 1)
        exclusions.setdefault(group, set()).add(roi_name)
    return exclusions


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        exclusions=_parse_exclusions(args.exclude),
        plot_range_um=(args.plot_min, args.plot_max),
        grid_step_um=args.grid_step,
        band=args.band,
    )
    print(result["metric_summary"].to_string(index=False))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
