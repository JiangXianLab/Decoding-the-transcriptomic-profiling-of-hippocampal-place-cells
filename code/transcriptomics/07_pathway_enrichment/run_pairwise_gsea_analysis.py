import contextlib
import io
import re
from pathlib import Path

import gseapy as gp
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omicverse as ov
import pandas as pd


# ===== User-adjustable settings =====
ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
DATE_TAG = "20260610"
INPUT_DIR = ROOT / "Data" / "results" / "PathwayEnrichment" / "GSEA"
GENESET_DIR = Path(r"F:\Code\Python\snRNAseq\genesets")
OUT_DIR = INPUT_DIR / f"pairwise_prerank_{DATE_TAG}"

PAIRWISE_COMPARISONS = {
    "place_vs_silent": INPUT_DIR / "place_silent_deseq2_20260128.csv",
    "place_vs_active_nonplace": INPUT_DIR / "place_nonplace_deseq2_20260128.csv",
    "active_nonplace_vs_silent": INPUT_DIR / "nonplace_silent_deseq2_20260128.csv",
}
COMPARISON_LABELS = {
    "place_vs_silent": "Place vs Silent",
    "place_vs_active_nonplace": "Place vs Active non-place",
    "active_nonplace_vs_silent": "Active non-place vs Silent",
}
GENESET_CONFIGS = {
    "GO_BP": {
        "filename": "GO_Biological_Process_2025.txt",
        "title": "GO biological process",
    },
    "KEGG": {
        "filename": "KEGG_2019_Mouse.txt",
        "title": "KEGG",
    },
    "Reactome": {
        "filename": "Reactome_Pathways_2024.txt",
        "title": "Reactome",
    },
}

RANK_COLUMN = "log2FoldChange"
RANK_DIRECTION_NOTE = "positive score indicates enrichment toward the first group in comparison"
MIN_SIZE = 5
MAX_SIZE = 1000
PERMUTATION_NUM = 1000
THREADS = 4
SEED = 6
FDR_CUTOFF = 0.25
TOP_TERM = 10
FIGURE_FORMATS = ("png", "svg", "pdf")
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def configure_matplotlib():
    with contextlib.redirect_stdout(io.StringIO()):
        ov.plot_set()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def validate_input_paths():
    required_paths = list(PAIRWISE_COMPARISONS.values())
    required_paths.extend(GENESET_DIR / config["filename"] for config in GENESET_CONFIGS.values())
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")


def load_ranked_genes(path, rank_column=RANK_COLUMN):
    df = pd.read_csv(path, index_col=0)
    if rank_column not in df.columns:
        raise ValueError(f"Ranking input must contain column {rank_column!r}: {path}")

    ranked = df[[rank_column]].reset_index()
    ranked.columns = ["gene", "score"]
    ranked["gene"] = ranked["gene"].astype(str)
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked = ranked.replace([np.inf, -np.inf], np.nan)
    ranked = ranked.dropna(subset=["gene", "score"])
    ranked = ranked[ranked["gene"].str.len() > 0]
    ranked = ranked.sort_values("score", ascending=False)
    ranked = ranked.drop_duplicates("gene", keep="first")
    if ranked.empty:
        raise ValueError(f"No ranked genes available after filtering: {path}")
    return ranked.reset_index(drop=True)


def prepare_genesets():
    genesets = {}
    for name, config in GENESET_CONFIGS.items():
        genesets[name] = ov.utils.geneset_prepare(
            str(GENESET_DIR / config["filename"]),
            organism="Mouse",
        )
    return genesets


def run_prerank_gsea(ranked, pathway_dict):
    pre_res = gp.prerank(
        rnk=ranked,
        gene_sets=pathway_dict,
        threads=THREADS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        permutation_num=PERMUTATION_NUM,
        outdir=None,
        seed=SEED,
        verbose=True,
    )
    if pre_res is None or pre_res.res2d is None:
        return pd.DataFrame()
    return pre_res.res2d.copy()


def comparison_output_dir(comparison):
    return OUT_DIR / comparison


def safe_name(value):
    value = SAFE_NAME_PATTERN.sub("_", str(value)).strip("_")
    return value or "gsea"


def save_gsea_result(comparison, geneset_name, result, source_path):
    out_dir = comparison_output_dir(comparison)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = result.copy()
    result.insert(0, "source_file", Path(source_path).name)
    result.insert(0, "rank_metric", RANK_COLUMN)
    result.insert(0, "geneset", geneset_name)
    result.insert(0, "comparison", comparison)
    out_path = out_dir / f"{comparison}_GSEA_{geneset_name}_results_{DATE_TAG}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def fdr_column(result):
    for column in ("FDR q-val", "FDR", "Adjusted P-value"):
        if column in result.columns:
            return column
    raise ValueError("GSEA result must contain an FDR column.")


def leading_gene_count(value):
    if pd.isna(value):
        return 1
    genes = [gene for gene in str(value).replace(",", ";").split(";") if gene.strip()]
    return max(len(genes), 1)


def select_dotplot_terms(result, top_term=TOP_TERM, fdr_cutoff=FDR_CUTOFF):
    if result is None or result.empty:
        return pd.DataFrame()
    if "Term" not in result.columns or "NES" not in result.columns:
        raise ValueError("GSEA result must contain 'Term' and 'NES' columns.")

    fdr_col = fdr_column(result)
    plot_df = result.copy()
    plot_df["NES"] = pd.to_numeric(plot_df["NES"], errors="coerce")
    plot_df[fdr_col] = pd.to_numeric(plot_df[fdr_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["Term", "NES", fdr_col])
    if plot_df.empty:
        return pd.DataFrame()

    significant = plot_df[plot_df[fdr_col] <= fdr_cutoff]
    if significant.empty:
        significant = plot_df
    selected = significant.assign(abs_nes=significant["NES"].abs())
    selected = selected.sort_values([fdr_col, "abs_nes"], ascending=[True, False]).head(top_term)
    return selected.drop(columns=["abs_nes"]).reset_index(drop=True)


def save_figure(fig, out_dir, figure_base):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for ext in FIGURE_FORMATS:
        out_path = out_dir / f"{figure_base}.{ext}"
        fig.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")
    plt.close(fig)
    return saved_paths


def plot_gsea_dotplot(comparison, geneset_name, result):
    selected = select_dotplot_terms(result)
    if selected.empty:
        print(f"Skipped dotplot: {comparison} {geneset_name} has no plottable terms.")
        return []

    fdr_col = fdr_column(selected)
    plot_df = selected.iloc[::-1].reset_index(drop=True)
    fdr = plot_df[fdr_col].astype(float).clip(lower=1e-300)
    minus_log10_fdr = -np.log10(fdr)
    if "Lead_genes" in plot_df.columns:
        sizes = plot_df["Lead_genes"].map(leading_gene_count).astype(float)
    else:
        sizes = pd.Series([1.0] * len(plot_df))
    sizes = 25 + sizes.clip(upper=20) * 9

    fig_height = max(2.3, 0.32 * len(plot_df) + 1.1)
    fig, ax = plt.subplots(figsize=(4.6, fig_height), dpi=300)
    y_positions = np.arange(len(plot_df))
    scatter = ax.scatter(
        plot_df["NES"].astype(float),
        y_positions,
        s=sizes,
        c=minus_log10_fdr,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
    )
    ax.axvline(0, color="#808080", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["Term"], fontsize=7)
    ax.set_xlabel("NES", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=3, width=0.8)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.set_title(
        f"{COMPARISON_LABELS.get(comparison, comparison)} - {GENESET_CONFIGS[geneset_name]['title']}",
        fontsize=8,
        pad=6,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(r"$-log_{10}$(FDR)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    figure_base = f"{comparison}_GSEA_{geneset_name}_dotplot_{DATE_TAG}"
    return save_figure(fig, comparison_output_dir(comparison), figure_base)


def summarize_result(comparison, geneset_name, result, source_path, ranked_count):
    fdr_col = fdr_column(result) if not result.empty else "FDR q-val"
    significant_count = 0
    if fdr_col in result.columns:
        fdr = pd.to_numeric(result[fdr_col], errors="coerce")
        significant_count = int((fdr <= FDR_CUTOFF).sum())
    return {
        "comparison": comparison,
        "comparison_label": COMPARISON_LABELS.get(comparison, comparison),
        "geneset": geneset_name,
        "source_file": Path(source_path).name,
        "rank_metric": RANK_COLUMN,
        "ranked_gene_count": ranked_count,
        "term_count": len(result),
        f"terms_fdr_le_{FDR_CUTOFF}": significant_count,
    }


def write_run_metadata(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pairwise_gsea_run_metadata_{DATE_TAG}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def main():
    configure_matplotlib()
    validate_input_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pathway_dicts = prepare_genesets()
    metadata_records = []

    print(f"Date tag: {DATE_TAG}")
    print(f"Rank metric: {RANK_COLUMN}")
    print(f"Rank direction: {RANK_DIRECTION_NOTE}")
    print(f"Output directory: {OUT_DIR}")

    for comparison, source_path in PAIRWISE_COMPARISONS.items():
        ranked = load_ranked_genes(source_path)
        print(
            f"{comparison}: ranked genes={len(ranked)}, "
            f"score range=({ranked['score'].min():.4g}, {ranked['score'].max():.4g})"
        )
        for geneset_name, pathway_dict in pathway_dicts.items():
            print(f"Running GSEA: {comparison} / {geneset_name}")
            result = run_prerank_gsea(ranked, pathway_dict)
            save_gsea_result(comparison, geneset_name, result, source_path)
            plot_gsea_dotplot(comparison, geneset_name, result)
            metadata_records.append(
                summarize_result(comparison, geneset_name, result, source_path, len(ranked))
            )

    write_run_metadata(metadata_records)


if __name__ == "__main__":
    main()
