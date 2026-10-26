"""
Detected genes and UMI count summaries.

Panel A:
    x: non-place / place / silent
    violin: all nuclei in each functional class
    points: mean per mouse per functional class

Panel B:
    x: 750 nm / 830 nm
    violin: all nuclei in each wavelength group
    points: mean per mouse per wavelength group

Statistics are Mann-Whitney U tests on mouse-level means.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import omicverse as ov
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import mannwhitneyu


INPUT_H5AD = r"F:\Data\snRNAseq\PlaceCell\FilteredData\PCAnalysis_all_QC_20260127.h5ad"


def add_ordered_display_column(df, source_col, display_col, labels, order):
    """Map internal group labels to ordered plotting labels."""
    plot_df = df[df[source_col].isin(order)].copy()
    display_order = [labels[group] for group in order]
    plot_df[display_col] = plot_df[source_col].map(labels)
    plot_df[display_col] = pd.Categorical(
        plot_df[display_col],
        categories=display_order,
        ordered=True,
    )
    return plot_df, display_order


def summarize_mouse_means(cell_df, group_col, display_col, labels, order, metric_col):
    """Mean metric value per mouse and group for points/statistics."""
    mouse_summary = (
        cell_df
        .groupby(["sample", group_col], observed=True)[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: "mean_metric"})
    )
    display_order = [labels[group] for group in order]
    mouse_summary[display_col] = mouse_summary[group_col].map(labels)
    mouse_summary[display_col] = pd.Categorical(
        mouse_summary[display_col],
        categories=display_order,
        ordered=True,
    )
    return mouse_summary


def export_metric_csvs(
    *,
    obs,
    metric_col,
    metric_output_name,
    group_col,
    source_col,
    display_col,
    labels,
    order,
    group_output_name,
    output_dir,
):
    """Export cell-level metric values and mouse-level means for one grouping."""
    cell_df = obs[["sample", source_col, metric_col]].copy()
    cell_df.insert(0, "cell_id", cell_df.index.astype(str))
    if source_col != group_col:
        cell_df = cell_df.rename(columns={source_col: group_col})
    if group_col == "wavelength":
        cell_df[group_col] = cell_df[group_col].astype(float)

    cell_df, _ = add_ordered_display_column(
        cell_df,
        source_col=group_col,
        display_col=display_col,
        labels=labels,
        order=order,
    )
    cell_df = cell_df.sort_values([display_col, "sample", "cell_id"])

    mean_col = f"mean_{metric_col}"
    mouse_mean_df = (
        cell_df
        .groupby(["sample", group_col, display_col], observed=True)
        .agg(
            n_cells=(metric_col, "size"),
            **{mean_col: (metric_col, "mean")},
        )
        .reset_index()
        .sort_values([display_col, "sample"])
    )

    cell_csv = output_dir / f"{metric_output_name}_per_cell_by_{group_output_name}.csv"
    mouse_mean_csv = output_dir / f"{metric_output_name}_mouse_mean_by_{group_output_name}.csv"
    cell_df.to_csv(cell_csv, index=False)
    mouse_mean_df.to_csv(mouse_mean_csv, index=False)

    print(f"\nSaved cell-level {metric_col} CSV: {cell_csv}")
    print(f"Saved mouse-level mean {metric_col} CSV: {mouse_mean_csv}")
    return cell_csv, mouse_mean_csv


def mann_whitney_mouse_mean_tests(mouse_summary, group_col, order):
    pair_results = []
    for c1, c2 in combinations(order, 2):
        vals1 = mouse_summary.loc[mouse_summary[group_col] == c1, "mean_metric"]
        vals2 = mouse_summary.loc[mouse_summary[group_col] == c2, "mean_metric"]
        if len(vals1) < 1 or len(vals2) < 1:
            pval = np.nan
        else:
            stat, pval = mannwhitneyu(vals1, vals2, alternative="two-sided")
        pair_results.append((c1, c2, pval))
    pair_results.sort(key=lambda x: np.inf if pd.isna(x[2]) else x[2])
    return pair_results


def pvalue_to_stars(pval):
    if pd.isna(pval):
        return "n/a"
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def plot_violin_with_mouse_means(
    *,
    cell_df,
    mouse_summary,
    metric_col,
    group_col,
    display_col,
    order,
    display_order,
    palette,
    title,
    xlabel,
    ylabel,
    output_stem,
    output_dir,
):
    """Violin shows all nuclei; overlaid points/statistics use mouse-level means."""
    if cell_df.empty:
        raise ValueError(f"No nuclei available for {title}.")

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(4.2, 4.8), dpi=300)

    sns.violinplot(
        data=cell_df,
        x=display_col,
        y=metric_col,
        order=display_order,
        palette=palette,
        inner="quartile",
        cut=0,
        linewidth=0.9,
        saturation=0.85,
        ax=ax,
    )

    sns.stripplot(
        data=mouse_summary,
        x=display_col,
        y="mean_metric",
        order=display_order,
        palette=palette,
        size=5.5,
        alpha=0.95,
        linewidth=0.55,
        edgecolor="black",
        jitter=0.12,
        ax=ax,
    )

    for idx, group in enumerate(order):
        vals = mouse_summary.loc[mouse_summary[group_col] == group, "mean_metric"]
        if vals.empty:
            continue

        sem_val = vals.sem()
        if pd.isna(sem_val):
            sem_val = 0

        ax.errorbar(
            idx,
            vals.mean(),
            yerr=sem_val,
            fmt="D",
            color="black",
            ecolor="black",
            markersize=3.6,
            elinewidth=1.4,
            capsize=5.5,
            capthick=1.4,
            zorder=10,
        )

    pair_results = mann_whitney_mouse_mean_tests(mouse_summary, group_col, order)
    y_data = cell_df[metric_col]
    y_max = y_data.max()
    y_min = y_data.min()
    y_range = y_max - y_min
    if pd.isna(y_range) or y_range == 0:
        y_range = max(abs(y_max) * 0.05, 1)
    y_top = y_max + y_range * 0.04
    h = y_range * 0.025
    group_to_x = {group: i for i, group in enumerate(order)}

    for i, (c1, c2, pval) in enumerate(pair_results):
        x1 = group_to_x[c1]
        x2 = group_to_x[c2]
        y = y_top + y_range * 0.075 * i
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8, clip_on=False)
        ax.text(
            (x1 + x2) / 2,
            y + h + y_range * 0.008,
            pvalue_to_stars(pval),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold", pad=9)
    ax.tick_params(axis="both", labelsize=9, length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(False)
    ax.set_ylim(y_min - y_range * 0.04, y_top + y_range * (0.12 + 0.08 * max(len(pair_results), 1)))

    fig.tight_layout()
    output_png = output_dir / f"{output_stem}.png"
    output_pdf = output_dir / f"{output_stem}.pdf"
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== {title}: cell-level counts used for violin plot ===")
    print(cell_df[display_col].value_counts().reindex(display_order).to_string())

    print(f"\n=== {title}: mouse-level means used for points and statistics ===")
    print(mouse_summary.sort_values([display_col, "sample"]).to_string(index=False))

    print(f"\n=== {title}: mouse-level Mann-Whitney U Test Results ===")
    label_lookup = dict(zip(order, display_order))
    for c1, c2, pval in pair_results:
        vals1 = mouse_summary.loc[mouse_summary[group_col] == c1, "mean_metric"]
        vals2 = mouse_summary.loc[mouse_summary[group_col] == c2, "mean_metric"]
        print(
            f"  {label_lookup[c1]} (n={len(vals1)}, mean+/-SEM={vals1.mean():.1f}+/-{vals1.sem():.1f}) vs "
            f"{label_lookup[c2]} (n={len(vals2)}, mean+/-SEM={vals2.mean():.1f}+/-{vals2.sem():.1f}): "
            f"p={pval:.2e}"
        )

    print(f"\nSaved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")
    return output_png, output_pdf


# --- Load & QC ---
adata_raw = sc.read_h5ad(INPUT_H5AD)
adata = adata_raw.copy()
adata = ov.pp.qc(
    adata,
    mt_startswith="mt-",
    tresh={"mito_perc": 0.05, "nUMIs": 1000, "detected_genes": 400},
    min_cells=3,
    min_genes=200,
    doublets=False,
)

output_dir = Path(__file__).resolve().parent

condition_order = ["nonplace", "place", "silent"]
condition_labels = {"nonplace": "non-place", "place": "place", "silent": "silent"}
condition_palette = {
    "non-place": "#E2A528",
    "place": "#6FA3C6",
    "silent": "#7F7F7F",
}

wavelength_order = [750.0, 830.0]
wavelength_labels = {750.0: "750 nm", 830.0: "830 nm"}
wavelength_palette = {
    "750 nm": "#C95F3F",
    "830 nm": "#6D6AA8",
}

plot_specs = [
    {
        "metric_col": "detected_genes",
        "metric_output_name": "detected_genes",
        "ylabel": "Detected genes per nucleus",
        "condition_title": "Panel A: Functional class summary",
        "condition_output": "panel_a_detected_genes_per_nucleus",
        "wavelength_title": "Panel B: Wavelength summary",
        "wavelength_output": "panel_b_detected_genes_by_wavelength",
    },
    {
        "metric_col": "nUMIs",
        "metric_output_name": "umis",
        "ylabel": "UMIs per nucleus",
        "condition_title": "Panel C: Functional class UMI summary",
        "condition_output": "panel_c_umis_by_functional_class",
        "wavelength_title": "Panel D: Wavelength UMI summary",
        "wavelength_output": "panel_d_umis_by_wavelength",
    },
]

for spec in plot_specs:
    metric_col = spec["metric_col"]
    metric_output_name = spec["metric_output_name"]

    export_metric_csvs(
        obs=adata.obs,
        metric_col=metric_col,
        metric_output_name=metric_output_name,
        group_col="functional_class",
        source_col="condition",
        display_col="display_functional_class",
        labels=condition_labels,
        order=condition_order,
        group_output_name="functional_class",
        output_dir=output_dir,
    )

    export_metric_csvs(
        obs=adata.obs,
        metric_col=metric_col,
        metric_output_name=metric_output_name,
        group_col="wavelength",
        source_col="property",
        display_col="display_wavelength",
        labels=wavelength_labels,
        order=wavelength_order,
        group_output_name="wavelength",
        output_dir=output_dir,
    )

    condition_cell_df = adata.obs[["sample", "condition", metric_col]].copy()
    condition_cell_df, condition_display_order = add_ordered_display_column(
        condition_cell_df,
        source_col="condition",
        display_col="display_condition",
        labels=condition_labels,
        order=condition_order,
    )
    condition_mouse_summary = summarize_mouse_means(
        condition_cell_df,
        group_col="condition",
        display_col="display_condition",
        labels=condition_labels,
        order=condition_order,
        metric_col=metric_col,
    )

    plot_violin_with_mouse_means(
        cell_df=condition_cell_df,
        mouse_summary=condition_mouse_summary,
        metric_col=metric_col,
        group_col="condition",
        display_col="display_condition",
        order=condition_order,
        display_order=condition_display_order,
        palette=condition_palette,
        title=spec["condition_title"],
        xlabel="Functional class",
        ylabel=spec["ylabel"],
        output_stem=spec["condition_output"],
        output_dir=output_dir,
    )

    wavelength_cell_df = adata.obs[["sample", "property", metric_col]].copy()
    wavelength_cell_df = wavelength_cell_df.rename(columns={"property": "wavelength"})
    wavelength_cell_df["wavelength"] = wavelength_cell_df["wavelength"].astype(float)
    wavelength_cell_df, wavelength_display_order = add_ordered_display_column(
        wavelength_cell_df,
        source_col="wavelength",
        display_col="display_wavelength",
        labels=wavelength_labels,
        order=wavelength_order,
    )
    wavelength_mouse_summary = summarize_mouse_means(
        wavelength_cell_df,
        group_col="wavelength",
        display_col="display_wavelength",
        labels=wavelength_labels,
        order=wavelength_order,
        metric_col=metric_col,
    )

    plot_violin_with_mouse_means(
        cell_df=wavelength_cell_df,
        mouse_summary=wavelength_mouse_summary,
        metric_col=metric_col,
        group_col="wavelength",
        display_col="display_wavelength",
        order=wavelength_order,
        display_order=wavelength_display_order,
        palette=wavelength_palette,
        title=spec["wavelength_title"],
        xlabel="Wavelength",
        ylabel=spec["ylabel"],
        output_stem=spec["wavelength_output"],
        output_dir=output_dir,
    )
