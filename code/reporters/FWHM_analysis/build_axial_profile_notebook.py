#!/usr/bin/env python
"""Build the reproducible A76 peak-aligned axial-profile notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "A76_peak_aligned_axial_profiles.ipynb"


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "fwhmz-gui",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# A76 peak-aligned axial fluorescence profiles

## tl;dr

- Each ROI is aligned to its own axial fluorescence peak (`0 µm`) before averaging.
- The solid line is the ROI-level mean; the shaded band is **±SEM across ROIs**.
- Current data contain 19 PAGFP/940 ROIs and 8 PAmCherry/1070 ROIs before any explicit exclusion.
- The PAmCherry mean has a broader positive-Z tail, but the individual-profile QC panel shows that a few broad or secondary-peak profiles contribute strongly to that shape.

This is a descriptive ROI-level comparison from A76. Animal-level inference requires biological replicates."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Key assumptions

1. `940` files are labeled **PAGFP** and `1070` files are labeled **PAmCherry**.
2. Profiles use the saved `smoothed_normalized_intensity` generated with Gaussian `sigma=1 plane`.
3. Every ROI is normalized and peak-aligned before averaging, so absolute brightness differences are intentionally removed.
4. QC flags are reported but do not silently remove data. Add ROI names to `EXCLUDE_ROIS` only after a documented decision.
5. `n` denotes ROIs, not mice. Double peaks should be classified separately if the final statistic is intended to represent single-target axial precision."""
        ),
        nbf.v4.new_code_cell(
            """%matplotlib inline
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path.cwd()
if not (ANALYSIS_DIR / "plot_peak_aligned_profiles.py").exists():
    ANALYSIS_DIR = Path.cwd() / "FWHM" / "analysis"
if not (ANALYSIS_DIR / "plot_peak_aligned_profiles.py").exists():
    raise FileNotFoundError("Run this notebook from its folder or the repository root.")
sys.path.insert(0, str(ANALYSIS_DIR))

from plot_peak_aligned_profiles import (
    plot_individual_profiles,
    plot_mean_profiles,
    run_analysis,
)

pd.set_option("display.max_colwidth", 100)"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """INPUT_DIR = Path(r"F:\\Figures\\PlaceCells\\Figure1\\Data\\FWHM\\A76\\Results")
OUTPUT_DIR = ANALYSIS_DIR / "results" / "A76"

# Explicit exclusions only. Example: {"PAGFP": {"13A"}, "PAmCherry": set()}
EXCLUDE_ROIS = {
    "PAGFP": set(),
    "PAmCherry": set(),
}

PLOT_RANGE_UM = (-50.0, 50.0)
GRID_STEP_UM = None  # None uses the smallest acquired Z step; currently 1 µm.
ERROR_BAND = "sem"  # Change to "sd" to show between-ROI spread.

result = run_analysis(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    prefix="A76",
    exclusions=EXCLUDE_ROIS,
    plot_range_um=PLOT_RANGE_UM,
    grid_step_um=GRID_STEP_UM,
    band=ERROR_BAND,
)

print(f"Loaded {len(result['records'])} ROI profiles")
print(f"Outputs: {OUTPUT_DIR.resolve()}")"""
        ),
        nbf.v4.new_markdown_cell(
            """### QC inventory

`included` is controlled only by `EXCLUDE_ROIS`. Inspect `qc_flags`, edge values, the original per-ROI PNG, and the individual-profile panel before excluding an ROI."""
        ),
        nbf.v4.new_code_cell(
            """qc_columns = [
    "group", "roi_name", "included", "qc_flags", "peak_z_um",
    "fwhm_z_um", "edge_left_normalized", "edge_right_normalized",
]
result["qc_table"][qc_columns]"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """metric_summary = result["metric_summary"].copy()
metric_summary.round(3)"""
        ),
        nbf.v4.new_markdown_cell(
            """### Individual peak-aligned profiles

Use this panel for QC. Broad shoulders or secondary peaks can materially change the group-average tail even after peak alignment."""
        ),
        nbf.v4.new_code_cell(
            """qc_fig = plot_individual_profiles(result["aligned"])
qc_fig"""
        ),
        nbf.v4.new_markdown_cell(
            """### Main panel: mean normalized axial profiles

The curve shape is based on all explicitly included ROIs. The ribbon is ±SEM across ROIs at each aligned Z position."""
        ),
        nbf.v4.new_code_cell(
            """main_fig = plot_mean_profiles(result["profile_summary"], band=ERROR_BAND)
main_fig"""
        ),
        nbf.v4.new_markdown_cell("### Exported files"),
        nbf.v4.new_code_cell(
            """for key, path in result["paths"].items():
    print(f"{key:24s} {Path(path).resolve()}")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

- With all current ROIs included, the saved ROI-level FWHM values are summarized separately from the averaged curve.
- The average profile is appropriate for displaying axial distribution after peak alignment; it is not itself an average FWHM measurement.
- The broader PAmCherry positive-Z tail should be reported descriptively until broad/double-peak ROIs are classified using a predefined rule.
- For the final manuscript, report the ribbon definition, ROI count, mouse count, exclusion criteria, Z step, and Gaussian sigma in the legend/methods."""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
