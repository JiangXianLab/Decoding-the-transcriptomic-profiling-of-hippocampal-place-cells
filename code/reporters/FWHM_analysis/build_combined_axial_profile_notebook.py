#!/usr/bin/env python
"""Build the combined A76+A82 peak-aligned axial-profile notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "A76_A82_peak_aligned_axial_profiles.ipynb"


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
            """# A76 + A82 peak-aligned axial fluorescence profiles

## tl;dr

- Each ROI is normalized and aligned to its own axial fluorescence peak (`0 um`).
- The main plot shows three conditions: 750 nm to PAGFP, 830 nm to PAGFP, and 830 nm to PAmCherry. Solid lines are means and ribbons are SEM across ROIs.
- Animal identity is retained in every QC/source table, and per-animal FWHM summaries are exported separately.
- The same user-specified ROI exclusions are applied to PAGFP and PAmCherry within each animal, and excluded rows remain listed in the QC table."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Key assumptions

1. `940` detection profiles are labeled PAGFP and `1070` detection profiles are labeled PAmCherry.
2. The saved `smoothed_normalized_intensity` is used; both animals were analyzed with a 1 um Z step and Gaussian sigma of 1 plane.
3. Peak-aligned source data are retained from `-50 to 50 um`; the main panel displays `-30 to 30 um` for a more compact comparison. Pointwise ROI counts decrease near the tails because peak shifting changes available coverage.
4. Exclusion is paired by ROI name within each animal: A76 excludes 17A/18A/19A and A82 excludes 6A/8A/14A/16A/19A/20A/23A from both reporters when present.
5. PAGFP-only ROI labels are assigned to the 750 nm group; labels with matched PAGFP and PAmCherry records are assigned to the two 830 nm groups. This inferred grouping must be checked against the stimulation log before publication.
6. The main curves are ROI-pooled descriptive results. With only two animals, the uncertainty ribbons must not be interpreted as animal-level inference."""
        ),
        nbf.v4.new_code_cell(
            """%matplotlib inline
from pathlib import Path
import sys

import pandas as pd

ANALYSIS_DIR = Path.cwd()
if not (ANALYSIS_DIR / "plot_peak_aligned_profiles.py").exists():
    ANALYSIS_DIR = Path.cwd() / "FWHM" / "analysis"
if not (ANALYSIS_DIR / "plot_peak_aligned_profiles.py").exists():
    raise FileNotFoundError("Run this notebook from its folder or the repository root.")
sys.path.insert(0, str(ANALYSIS_DIR))

from plot_peak_aligned_profiles import (
    plot_individual_profiles,
    run_multi_animal_analysis,
)
from plot_three_condition_axial_profiles import (
    plot_profiles as plot_three_condition_profiles,
    run as run_three_condition_profiles,
)

pd.set_option("display.max_colwidth", 100)"""
        ),
        nbf.v4.new_markdown_cell("## Data and parameters"),
        nbf.v4.new_code_cell(
            """INPUT_DIRS = {
    "A76": Path(r"F:\\Figures\\PlaceCells\\Figure1\\Data\\FWHM\\A76\\Results"),
    "A82": Path(r"F:\\Figures\\PlaceCells\\Figure1\\Data\\FWHM\\A82\\results"),
}
OUTPUT_DIR = ANALYSIS_DIR / "results" / "A76_A82_user_exclusions"

# Apply the same ROI-name exclusions to both reporters within each animal.
EXCLUDE_ROIS = {
    "A76": {
        "PAGFP": {"17A", "18A", "19A"},
        "PAmCherry": {"17A", "18A", "19A"},
    },
    "A82": {
        "PAGFP": {"6A", "8A", "14A", "16A", "19A", "20A", "23A"},
        "PAmCherry": {"6A", "8A", "14A", "16A", "19A", "20A", "23A"},
    },
}

PLOT_RANGE_UM = (-50.0, 50.0)
GRID_STEP_UM = None  # Uses the acquired 1 um Z step.
ERROR_BAND = "sem"

result = run_multi_animal_analysis(
    input_dirs=INPUT_DIRS,
    output_dir=OUTPUT_DIR,
    prefix="A76_A82_user_exclusions",
    exclusions=EXCLUDE_ROIS,
    plot_range_um=PLOT_RANGE_UM,
    grid_step_um=GRID_STEP_UM,
    band=ERROR_BAND,
)

three_profile_result = run_three_condition_profiles(
    input_csv=result["paths"]["aligned_csv"],
    output_dir=OUTPUT_DIR / "three_condition",
    prefix="A76_A82_750_830_three_condition_axial_profile",
)

print(f"Loaded {len(result['records'])} ROI profiles from {len(INPUT_DIRS)} animals")
print(f"Outputs: {OUTPUT_DIR.resolve()}")
condition_counts = (
    three_profile_result["assigned"]
    .groupby("condition")["roi_key"]
    .nunique()
)
display(condition_counts.rename("n_roi"))"""
        ),
        nbf.v4.new_markdown_cell(
            """## QC inventory

`included` is controlled only by `EXCLUDE_ROIS`. Excluded rows remain visible here for traceability."""
        ),
        nbf.v4.new_code_cell(
            """qc_columns = [
    "animal_id", "group", "roi_name", "included", "qc_flags",
    "peak_z_um", "fwhm_z_um", "edge_left_normalized", "edge_right_normalized",
]
result["qc_table"][qc_columns]"""
        ),
        nbf.v4.new_markdown_cell("## FWHM summaries"),
        nbf.v4.new_code_cell(
            """print("Pooled ROI summary")
display(result["metric_summary"].round(3))
print("Per-animal summary")
display(result["animal_metric_summary"].round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Individual peak-aligned profiles

This is the primary QC view for identifying broad shoulders and candidate secondary peaks."""
        ),
        nbf.v4.new_code_cell(
            """qc_fig = plot_individual_profiles(result["aligned"])
qc_fig"""
        ),
        nbf.v4.new_markdown_cell(
            """## Main panel: three activation/reporter conditions

The curves pool retained ROIs from both animals. PAGFP is separated into inferred 750 nm and 830 nm groups, while PAmCherry represents the matched 830 nm group. Pointwise `n_roi` values used for the SEM ribbons are available in the exported summary table."""
        ),
        nbf.v4.new_code_cell(
            """main_fig = plot_three_condition_profiles(three_profile_result["summary"])
main_fig"""
        ),
        nbf.v4.new_markdown_cell("## Exported files"),
        nbf.v4.new_code_cell(
            """print("Core QC outputs")
for key, path in result["paths"].items():
    print(f"{key:28s} {Path(path).resolve()}")

print("\\nThree-condition figure outputs")
for key, path in three_profile_result["paths"].items():
    print(f"{key:28s} {Path(path).resolve()}")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

- Report both ROI counts and animal counts in the manuscript legend.
- The three-condition main profile and FWHM summary use the same explicitly retained profiles and grouping logic.
- The exclusion rule is paired across reporters within each animal, rather than selected independently from each channel's curve shape.
- Do not infer a wavelength or reporter mechanism from the pooled curve alone; reporter properties, expression, saturation, tissue depth, and multi-target activation remain alternative explanations."""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
