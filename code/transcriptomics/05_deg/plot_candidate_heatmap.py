from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# ===== User-adjustable settings =====
ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
H5AD_PATH = Path(r"F:\Data\snRNAseq\PlaceCell\FilteredData\PCAnalysis_all_QC_20260127.h5ad")
ANNOTATION_PATH = ROOT / "Data" / "FilteredData" / "All_QC_filtered_annotation_20260604.h5ad"
CANDIDATE_PATH = ROOT / "results" / "activity_candidate_final_genes_20260612.csv"
DEG_PATH = (
    ROOT
    / "Data"
    / "DEGs"
    / "SourceData_DEG_ActiveNonPlace_vs_Silent_Wilcoxon_CA1ProsOnly.csv"
)
CELL_TYPE = "CA1-ProS"
RUN_TAG = "CA1_ProS_activity_20260612"
OUT_DIR = (
    ROOT
    / "Data"
    / "results"
    / "DEGs"
    / "FindMarkers_wilcoxon"
    / "Figures"
    / "PN"
)
CELL_OUT_BASE = OUT_DIR / f"activity_candidate_heatmap_cell_lognorm_{RUN_TAG}"
CELL_MATRIX_PATH = OUT_DIR / f"activity_candidate_heatmap_cell_lognorm_matrix_{RUN_TAG}.csv"
SAMPLE_OUT_BASE = OUT_DIR / f"activity_candidate_heatmap_sample_condition_lognorm_{RUN_TAG}"
SAMPLE_MATRIX_PATH = OUT_DIR / f"activity_candidate_heatmap_sample_condition_lognorm_matrix_{RUN_TAG}.csv"
SAMPLE_CENTERED_OUT_BASE = OUT_DIR / f"activity_candidate_heatmap_sample_condition_lognorm_centered_{RUN_TAG}"
SAMPLE_CENTERED_MATRIX_PATH = (
    OUT_DIR / f"activity_candidate_heatmap_sample_condition_lognorm_centered_matrix_{RUN_TAG}.csv"
)

CONDITION_ORDER = ["place", "nonplace", "silent"]
CONDITION_COLORS = {"place": "#6FA3C6", "nonplace": "#E2A528", "silent": "#7F7F7F"}
NORMALIZE_TARGET_SUM = 1e4
FIG_WIDTH = 4.0
FIG_HEIGHT_PER_GENE = 0.15
FIG_HEIGHT_PAD = 1.2
SAMPLE_FIG_WIDTH = 9.0
SAMPLE_FIG_HEIGHT_PER_GENE = 0.28
SAMPLE_FIG_HEIGHT_PAD = 2.0
CELL_FIG_WIDTH = 9.0
CELL_FIG_HEIGHT_PER_GENE = 0.16
CELL_FIG_HEIGHT_PAD = 1.3
GENE_LABEL_FONT_SIZE = 8
SAMPLE_LABEL_FONT_SIZE = 8
TITLE_FONT_SIZE = 10
COLORBAR_FONT_SIZE = 8
COLORBAR_TICK_FONT_SIZE = 7
SEPARATOR_LINE_WIDTH = 0.8
HEATMAP_COLORS = ["#22456F", "#528FAC", "#AECAD8", "#F2F2F2", "#F7D29D", "#EEB84B", "#E87A2D"]
CMAP = LinearSegmentedColormap.from_list("placecell_blue_orange", HEATMAP_COLORS, N=256)
ZSCORE_VMIN = -2
ZSCORE_VMAX = 2


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",  # keep text editable in Illustrator
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def load_candidate_genes():
    candidates = pd.read_csv(CANDIDATE_PATH)
    if "gene" not in candidates.columns:
        raise ValueError(f"Candidate CSV must contain a 'gene' column: {CANDIDATE_PATH}")

    genes = candidates["gene"].dropna().astype(str).drop_duplicates().tolist()
    if not genes:
        raise ValueError(f"No candidate genes found in: {CANDIDATE_PATH}")

    return genes


def load_candidate_table():
    candidates = pd.read_csv(CANDIDATE_PATH)
    required_cols = {"gene", "label"}
    missing_cols = required_cols.difference(candidates.columns)
    if missing_cols:
        raise ValueError(f"Candidate CSV missing columns {sorted(missing_cols)}: {CANDIDATE_PATH}")

    candidates = candidates[["gene", "label"]].dropna(subset=["gene"]).copy()
    candidates["_source_order"] = np.arange(len(candidates))
    candidates["gene"] = candidates["gene"].astype(str)
    candidates["label"] = candidates["label"].astype(str)
    candidates = candidates.drop_duplicates(subset="gene", keep="first")

    deg = pd.read_csv(DEG_PATH, index_col=0)
    if "avg_log2FC" not in deg.columns:
        raise ValueError(f"DEG CSV must contain an 'avg_log2FC' column: {DEG_PATH}")
    deg.index = deg.index.astype(str)

    candidates = candidates.merge(
        deg[["avg_log2FC"]],
        left_on="gene",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    missing_deg = candidates.loc[candidates["avg_log2FC"].isna(), "gene"].tolist()
    if missing_deg:
        print(
            "Warning: place-nonplace DEG table does not cover all candidates; "
            f"falling back to source order for {len(missing_deg)} genes."
        )

    candidates["abs_avg_log2FC"] = candidates["avg_log2FC"].abs().fillna(-1)
    label_order = pd.CategoricalDtype(categories=["Up", "Down"], ordered=True)
    candidates["_label_order"] = candidates["label"].astype(label_order)
    candidates = candidates.sort_values(
        ["_label_order", "abs_avg_log2FC", "_source_order"],
        ascending=[True, False, True],
    )
    candidates = candidates.drop(columns=["_source_order", "_label_order"])
    if candidates.empty:
        raise ValueError(f"No candidate genes found in: {CANDIDATE_PATH}")

    return candidates


def load_adata():
    adata = sc.read_h5ad(H5AD_PATH)
    adata_annotation = sc.read_h5ad(ANNOTATION_PATH)
    assert adata.n_obs == adata_annotation.n_obs
    assert (adata.obs_names == adata_annotation.obs_names).all()

    if "cell_types" not in adata_annotation.obs.columns:
        raise ValueError(f"Annotation H5AD obs must contain a 'cell_types' column: {ANNOTATION_PATH}")
    adata.obs["cell_types"] = adata_annotation.obs["cell_types"].values
    adata = adata[adata.obs["cell_types"] == CELL_TYPE].copy()

    if "condition" not in adata.obs.columns:
        raise ValueError(f"H5AD obs must contain a 'condition' column: {H5AD_PATH}")

    return adata


def normalize_adata_for_expression(adata):
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=NORMALIZE_TARGET_SUM)
    sc.pp.log1p(adata_norm)
    return adata_norm


def make_zscore_adata(adata, genes):
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Missing genes in adata.var_names ({len(missing)}): {missing}")

    adata_z = adata[:, genes].copy()
    X = adata_z.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    adata_z.layers["zscore"] = (X - mu) / sd

    adata_z.obs["condition"] = pd.Categorical(
        adata_z.obs["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    return adata_z


def make_sample_condition_mean_matrix(adata, genes):
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Missing genes in adata.var_names ({len(missing)}): {missing}")
    if "sample" not in adata.obs.columns:
        raise ValueError(f"H5AD obs must contain a 'sample' column: {H5AD_PATH}")

    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    obs = adata.obs[["sample", "condition"]].copy()
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=genes)
    mean_expr = expr_df.groupby([obs["sample"].astype(str), obs["condition"].astype(str)]).mean()
    mean_expr.index.names = ["sample", "condition"]

    mean_expr = mean_expr.reset_index()
    mean_expr["condition"] = pd.Categorical(
        mean_expr["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    mean_expr = mean_expr.sort_values(["condition", "sample"]).set_index(["sample", "condition"])

    plot_df = mean_expr.T.reindex(genes)
    return plot_df


def make_sample_condition_matrix(adata, candidates):
    genes = candidates["gene"].tolist()
    plot_df = make_sample_condition_mean_matrix(adata, genes)
    z = plot_df.sub(plot_df.mean(axis=1), axis=0)
    sd = plot_df.std(axis=1, ddof=0).replace(0, 1)
    z = z.div(sd, axis=0)
    z.index = pd.MultiIndex.from_frame(candidates[["label", "gene"]])
    return z


def make_sample_condition_centered_matrix(adata, candidates):
    genes = candidates["gene"].tolist()
    plot_df = make_sample_condition_mean_matrix(adata, genes)

    sample_baseline = plot_df.T.groupby(level="sample").transform("mean").T
    centered = plot_df - sample_baseline

    z = centered.sub(centered.mean(axis=1), axis=0)
    sd = centered.std(axis=1, ddof=0).replace(0, 1)
    z = z.div(sd, axis=0)
    z.index = pd.MultiIndex.from_frame(candidates[["label", "gene"]])
    return z


def make_cell_matrix(adata, candidates):
    genes = candidates["gene"].tolist()
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Missing genes in adata.var_names ({len(missing)}): {missing}")

    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    obs = adata.obs[["sample", "condition"]].copy()
    obs["condition"] = pd.Categorical(obs["condition"], categories=CONDITION_ORDER, ordered=True)
    obs["cell"] = obs.index.astype(str)
    obs = obs.sort_values(["condition", "sample", "cell"])

    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=genes).loc[obs.index]
    plot_df = expr_df.T.reindex(genes)
    z = plot_df.sub(plot_df.mean(axis=1), axis=0)
    sd = plot_df.std(axis=1, ddof=0).replace(0, 1)
    z = z.div(sd, axis=0)
    z.index = pd.MultiIndex.from_frame(candidates[["label", "gene"]])
    z.columns = pd.MultiIndex.from_frame(obs[["condition", "sample", "cell"]])
    return z


def plot_cell_heatmap(z, title="Activity candidate genes, log-normalized single cells"):
    labels = z.index.get_level_values("label")
    genes = z.index.get_level_values("gene")
    values = z.to_numpy()
    columns = list(z.columns)
    conditions = np.asarray([str(condition) for condition, _, _ in columns])
    condition_rgb = [mpl.colors.to_rgb(CONDITION_COLORS[condition]) for condition in conditions]

    fig_height = CELL_FIG_HEIGHT_PER_GENE * len(genes) + CELL_FIG_HEIGHT_PAD
    fig, (ax, ax_strip) = plt.subplots(
        2,
        1,
        figsize=(CELL_FIG_WIDTH, fig_height),
        gridspec_kw={"height_ratios": [len(genes), 0.8], "hspace": 0.02},
        constrained_layout=False,
    )

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=CMAP,
        norm=TwoSlopeNorm(vmin=ZSCORE_VMIN, vcenter=0, vmax=ZSCORE_VMAX),
    )
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=GENE_LABEL_FONT_SIZE)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=7)

    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color("#C58618" if label == "Up" else "#4F8DB3")

    label_values = labels.to_numpy()
    split_positions = np.where(label_values[:-1] != label_values[1:])[0]
    for pos in split_positions:
        ax.axhline(pos + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)

    condition_splits = np.where(conditions[:-1] != conditions[1:])[0]
    condition_centers = []
    condition_names = []
    start = 0
    for pos in list(condition_splits) + [len(conditions) - 1]:
        end = pos
        condition_centers.append((start + end) / 2)
        condition_names.append(conditions[start])
        ax.axvline(end + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)
        ax_strip.axvline(end + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)
        start = end + 1

    strip = np.asarray(condition_rgb, dtype=float)[None, :, :]
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_xticks(condition_centers)
    ax_strip.set_xticklabels(condition_names, fontsize=SAMPLE_LABEL_FONT_SIZE)
    ax_strip.tick_params(axis="x", length=0, pad=2)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=[ax, ax_strip], fraction=0.026, pad=0.02)
    cbar.set_label("z-score", fontsize=COLORBAR_FONT_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, length=2)

    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.08, top=0.94)
    return fig


def plot_sample_condition_heatmap(z, title="Activity candidate genes, sample-condition mean"):
    labels = z.index.get_level_values("label")
    genes = z.index.get_level_values("gene")
    values = z.to_numpy()
    columns = list(z.columns)
    column_labels = [f"{sample}\n{condition}" for sample, condition in columns]
    condition_rgb = [mpl.colors.to_rgb(CONDITION_COLORS[str(condition)]) for _, condition in columns]

    fig_height = SAMPLE_FIG_HEIGHT_PER_GENE * len(genes) + SAMPLE_FIG_HEIGHT_PAD
    fig, (ax, ax_strip) = plt.subplots(
        2,
        1,
        figsize=(SAMPLE_FIG_WIDTH, fig_height),
        gridspec_kw={"height_ratios": [len(genes), 1.0], "hspace": 0.02},
        constrained_layout=False,
    )

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=CMAP,
        norm=TwoSlopeNorm(vmin=ZSCORE_VMIN, vcenter=0, vmax=ZSCORE_VMAX),
    )
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=GENE_LABEL_FONT_SIZE)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=7)

    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color("#C58618" if label == "Up" else "#4F8DB3")

    label_values = labels.to_numpy()
    split_positions = np.where(label_values[:-1] != label_values[1:])[0]
    for pos in split_positions:
        ax.axhline(pos + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)

    conditions = np.asarray([str(condition) for _, condition in columns])
    condition_splits = np.where(conditions[:-1] != conditions[1:])[0]
    for pos in condition_splits:
        ax.axvline(pos + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)
        ax_strip.axvline(pos + 0.5, color="black", lw=SEPARATOR_LINE_WIDTH)

    strip = np.asarray(condition_rgb, dtype=float)[None, :, :]
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_xticks(np.arange(len(column_labels)))
    ax_strip.set_xticklabels(
        column_labels,
        rotation=90,
        fontsize=SAMPLE_LABEL_FONT_SIZE,
        ha="center",
        va="top",
    )
    ax_strip.tick_params(axis="x", length=0, pad=2)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=[ax, ax_strip], fraction=0.028, pad=0.03)
    cbar.set_label("z-score", fontsize=COLORBAR_FONT_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, length=2)

    fig.subplots_adjust(left=0.22, right=0.86, bottom=0.16, top=0.94)
    return fig


def save_figure(fig, out_base):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for ext in ["png", "svg", "pdf"]:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main():
    configure_matplotlib()
    candidates = load_candidate_table()
    genes = candidates["gene"].tolist()
    adata = load_adata()
    missing = [gene for gene in genes if gene not in adata.var_names]
    print(f"Candidate gene count: {len(genes)}")
    print(f"Missing gene count: {len(missing)}")
    print(f"Cell-level normalization: normalize_total target_sum={NORMALIZE_TARGET_SUM:g}, then log1p")

    adata_norm = normalize_adata_for_expression(adata)

    cell_z = make_cell_matrix(adata_norm, candidates)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_z.to_csv(CELL_MATRIX_PATH)
    cell_fig = plot_cell_heatmap(cell_z, title=f"{CELL_TYPE} activity candidate genes, log-normalized single cells")
    cell_saved_paths = save_figure(cell_fig, CELL_OUT_BASE)

    sample_z = make_sample_condition_matrix(adata_norm, candidates)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_z.to_csv(SAMPLE_MATRIX_PATH)
    sample_fig = plot_sample_condition_heatmap(
        sample_z,
        title=f"{CELL_TYPE} activity candidate genes, log-normalized sample-condition mean",
    )
    sample_saved_paths = save_figure(sample_fig, SAMPLE_OUT_BASE)

    sample_centered_z = make_sample_condition_centered_matrix(adata_norm, candidates)
    sample_centered_z.to_csv(SAMPLE_CENTERED_MATRIX_PATH)
    sample_centered_fig = plot_sample_condition_heatmap(
        sample_centered_z,
        title=f"{CELL_TYPE} activity candidate genes, log-normalized sample-centered contrast",
    )
    sample_centered_saved_paths = save_figure(sample_centered_fig, SAMPLE_CENTERED_OUT_BASE)

    print(f"Cell columns: {cell_z.shape[1]}")
    print(f"Saved cell matrix: {CELL_MATRIX_PATH}")
    for path in cell_saved_paths:
        print(f"Saved: {path}")
    print(f"Sample-condition columns: {sample_z.shape[1]}")
    print(f"Saved matrix: {SAMPLE_MATRIX_PATH}")
    for path in sample_saved_paths:
        print(f"Saved sample-condition: {path}")
    print(f"Centered sample-condition columns: {sample_centered_z.shape[1]}")
    print(f"Saved centered matrix: {SAMPLE_CENTERED_MATRIX_PATH}")
    for path in sample_centered_saved_paths:
        print(f"Saved centered sample-condition: {path}")


if __name__ == "__main__":
    main()
