import contextlib
import io
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import omicverse as ov
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


# ===== User-adjustable settings =====
ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
CANDIDATE_PATH = ROOT / "results" / "spatial_candidate_final_genes_20260609.csv"
H5AD_PATH = Path(r"F:\Data\snRNAseq\PlaceCell\FilteredData\PCAnalysis_all_QC_20260127.h5ad")
ANNOTATION_H5AD_PATH = ROOT / "Data" / "FilteredData" / "All_QC_filtered_annotation_20260604.h5ad"
GENESET_DIR = Path(r"F:\Code\Python\snRNAseq\genesets")
OUT_DIR = (
    ROOT
    / "Data"
    / "results"
    / "PathwayEnrichment"
    / "Spatial_Selectivity"
    / "CA1-ProS"
    / "genes104_mincell1_20260609"
)
DATE_TAG = "20260609"
CELL_TYPE_FILTER = "CA1-ProS"
BACKGROUND_MIN_CELLS = 1
EXPECTED_CANDIDATE_COUNT = 104

GENESET_CONFIGS = {
    "GO_BP": {
        "filename": "GO_Biological_Process_2025.txt",
        "title": "GO biological process",
        "figure_base": f"geneset_GO_Biological_Process_{DATE_TAG}",
    },
    "GO_MF": {
        "filename": "GO_Molecular_Function_2025.txt",
        "title": "GO molecular function",
        "figure_base": f"geneset_GO_Molecular_Function_{DATE_TAG}",
    },
    "GO_CC": {
        "filename": "GO_Cellular_Component_2025.txt",
        "title": "GO cellular component",
        "figure_base": f"geneset_GO_Cellular_Component_{DATE_TAG}",
    },
    "KEGG": {
        "filename": "KEGG_2019_Mouse.txt",
        "title": "KEGG",
        "figure_base": f"geneset_KEGG_{DATE_TAG}",
    },
    "Reactome": {
        "filename": "Reactome_Pathways_2024.txt",
        "title": "Reactome",
        "figure_base": f"geneset_Reactome_{DATE_TAG}",
    },
}

GO_MULTI_FIGURE_BASE = f"geneset_GO_BP_MF_CC_{DATE_TAG}"
GO_FIGURE_ORDER = ("GO_BP", "GO_CC")
CURATED_GO_FIGURE_BASE = f"geneset_GO_curated_BP_CC_{DATE_TAG}"
RESULT_PREFIX = f"spatial_selectivity_{EXPECTED_CANDIDATE_COUNT}_genes"
CURATED_GO_SOURCE_CSV = f"{RESULT_PREFIX}_GO_curated_terms_{DATE_TAG}.csv"
FIGURE_FORMATS = ("png", "svg", "pdf")
ENRICHR_REPORT_DIR = OUT_DIR / "enrichr_reports"
GO_ID_PATTERN = re.compile(r"\s*\(GO:\d+\)\s*$")
CURATED_GO_BAR_COLOR = "#7F7F7F"

CURATED_GO_MAX_TERMS = {"GO_BP": 6, "GO_CC": 8}
CURATED_GO_MIN_OVERLAP = 3
CURATED_GO_FDR_CUTOFF = 0.05
CURATED_GO_AXIS_WIDTH_SCALE = 1 / 3
CURATED_GO_INTER_PANEL_GAP_INCHES = 0.35
CURATED_GO_EXCLUDE_PATTERNS = (
    "intracellular membrane-bounded organelle",
    "intracellular membraneless organelle",
    "intracellular organelle",
    "membrane-bounded organelle",
    "membraneless organelle",
    "nucleus",
    "nuclear membrane",
    "nuclear lumen",
    "nucleolus",
    "chromatin",
    "chromosome",
    "spindle",
    "nf-kappab",
    "jnk",
    "jun kinase",
    "apoptotic",
    "catabolic process",
    "cilium",
    "azurophil",
    "ficolin-1-rich granule",
    "ribosomal subunit",
)
CURATED_GO_PRIORITY_TERMS = {
    "GO_BP": (
        "Regulation of Proteolysis",
        "Negative Regulation of Intracellular Signal Transduction",
        "Regulation of Microtubule Cytoskeleton Organization",
        "Receptor-Mediated Endocytosis",
        "Central Nervous System Development",
        "Cholesterol Homeostasis",
        "Sterol Homeostasis",
        "Negative Regulation of Protein Localization",
    ),
    "GO_CC": (
        "Endoplasmic Reticulum Membrane",
        "Lysosomal Membrane",
        "Microtubule Cytoskeleton",
        "Neuron Projection",
        "Lipid Droplet",
        "Late Endosome Membrane",
        "Cytoplasmic Vesicle Lumen",
        "Membrane Raft",
        "Endosome Membrane",
    ),
}
CURATED_GO_PANEL_TITLES = {
    "GO_BP": "GO biological processes associated with spatial selectivity",
    "GO_CC": "GO cellular components associated with spatial selectivity",
}

HEATMAP_COLORS = [
    "#22456F",
    "#528FAC",
    "#AECAD8",
    "#F2F2F2",
    "#F7D29D",
    "#EEB84B",
    "#E87A2D",
]
CMAP_CUSTOM = LinearSegmentedColormap.from_list("placecell_blue_orange", HEATMAP_COLORS, N=256)


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
        }
    )


def load_candidate_genes(path=CANDIDATE_PATH, expected_count=EXPECTED_CANDIDATE_COUNT):
    candidates = pd.read_csv(path)
    if "gene" not in candidates.columns:
        raise ValueError(f"Candidate CSV must contain a 'gene' column: {path}")

    genes = candidates["gene"].dropna().astype(str).drop_duplicates().tolist()
    if expected_count is not None and len(genes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique candidate genes, found {len(genes)}: {path}"
        )
    if not genes:
        raise ValueError(f"No candidate genes found in: {path}")

    return genes


def validate_input_paths():
    required_paths = [CANDIDATE_PATH, H5AD_PATH, ANNOTATION_H5AD_PATH]
    required_paths.extend(GENESET_DIR / config["filename"] for config in GENESET_CONFIGS.values())

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")


def load_adata_and_background_genes():
    adata = sc.read_h5ad(H5AD_PATH)
    adata_annotation = sc.read_h5ad(ANNOTATION_H5AD_PATH)
    if adata.n_obs != adata_annotation.n_obs:
        raise ValueError(
            f"adata and annotation cell counts differ: {adata.n_obs} != {adata_annotation.n_obs}"
        )
    if not (adata.obs_names == adata_annotation.obs_names).all():
        raise ValueError("adata and annotation obs_names are not identical.")
    if "cell_types" not in adata_annotation.obs.columns:
        raise ValueError(f"Annotation h5ad must contain obs['cell_types']: {ANNOTATION_H5AD_PATH}")

    adata.obs["cell_types"] = adata_annotation.obs["cell_types"].values
    adata = adata[adata.obs["cell_types"] == CELL_TYPE_FILTER].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No cells found for CELL_TYPE_FILTER={CELL_TYPE_FILTER!r}.")

    adata_bg = adata.copy()
    sc.pp.filter_genes(adata_bg, min_cells=BACKGROUND_MIN_CELLS)
    bg = adata_bg.var_names.astype(str).tolist()
    return adata, bg


def find_missing_genes(adata, genes):
    var_names = set(adata.var_names.astype(str))
    return [gene for gene in genes if gene not in var_names]


def run_geneset_enrichment(name, genes, background):
    config = GENESET_CONFIGS[name]
    pathway_dict = ov.utils.geneset_prepare(
        str(GENESET_DIR / config["filename"]),
        organism="Mouse",
    )

    result = ov.bulk.geneset_enrichment(
        gene_list=genes,
        pathways_dict=pathway_dict,
        pvalue_type="auto",
        background=background,
        organism="mouse",
        outdir=str(ENRICHR_REPORT_DIR),
        cutoff=0.25,
    )
    if result is None:
        return pd.DataFrame()
    return result


def result_csv_path(name):
    return OUT_DIR / f"{RESULT_PREFIX}_{name}_enrichment_{DATE_TAG}.csv"


def save_enrichment_result(name, result):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = result_csv_path(name)
    result.to_csv(out_path, index=False)
    print(f"{name} enrichment term count: {len(result)}")
    print(f"Saved: {out_path}")
    return out_path


def has_plot_terms(result):
    return isinstance(result, pd.DataFrame) and not result.empty and "logp" in result.columns


def get_logp_ticks(result):
    logp_min = result["logp"].min()
    logp_max = result["logp"].max()
    if pd.isna(logp_min) or pd.isna(logp_max):
        return [0, 1]
    if logp_min == logp_max:
        return [logp_min]
    return [logp_min, logp_max]


def save_figure(fig, figure_base):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for ext in FIGURE_FORMATS:
        out_path = OUT_DIR / f"{figure_base}.{ext}"
        fig.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")
    plt.close(fig)
    return saved_paths


def plot_single_enrichment(name, result):
    if not has_plot_terms(result):
        print(f"Skipped {name} plot: no enriched terms.")
        return []

    config = GENESET_CONFIGS[name]
    ax = ov.bulk.geneset_plot(
        result,
        figsize=(2, 4),
        fig_title=config["title"],
        cax_loc=[1.25, 0.45, 0.5, 0.02],
        bbox_to_anchor_used=(0, -15),
        node_diameter=40,
        node_size=[5, 10],
        text_knock=7,
        cmap=CMAP_CUSTOM,
        custom_ticks=get_logp_ticks(result),
    )

    if ax.collections and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.mappable.set_clim(result["logp"].min(), result["logp"].max())

    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", labelleft=True, labelright=False)

    return save_figure(ax.figure, config["figure_base"])


def plot_go_enrichment_figures(results):
    saved_paths = []
    for name in GO_FIGURE_ORDER:
        saved_paths.extend(plot_single_enrichment(name, results.get(name)))
    return saved_paths


def plot_go_multi(results):
    go_results = {
        "BP": results.get("GO_BP"),
        "MF": results.get("GO_MF"),
        "CC": results.get("GO_CC"),
    }
    go_results = {name: result for name, result in go_results.items() if has_plot_terms(result)}
    if not go_results:
        print("Skipped GO BP/MF/CC plot: no enriched GO terms.")
        return []

    colors_dict = {}
    if "BP" in go_results:
        colors_dict["BP"] = ov.pl.red_color[1]
    if "MF" in go_results:
        colors_dict["MF"] = ov.pl.green_color[1]
    if "CC" in go_results:
        colors_dict["CC"] = ov.pl.blue_color[1]

    try:
        ov.bulk.geneset_plot_multi(
            go_results,
            colors_dict,
            num=10,
            figsize=(8, 16),
            text_knock=20,
            fontsize=15,
            cmap=CMAP_CUSTOM,
        )
        fig = plt.gcf()
        return save_figure(fig, GO_MULTI_FIGURE_BASE)
    except ValueError as exc:
        plt.close("all")
        print(f"GO BP/MF/CC OmicVerse plot failed ({exc}); using fallback dot plot.")
        return plot_go_multi_fallback(go_results)


def plot_go_multi_fallback(go_results):
    frames = []
    for ontology, result in go_results.items():
        df = result.copy()
        if "Term" not in df.columns:
            continue
        df["ontology"] = ontology
        if "fraction" not in df.columns:
            if "num" in df.columns:
                df["fraction"] = df["num"] / EXPECTED_CANDIDATE_COUNT
            else:
                df["fraction"] = 0
        if "num" not in df.columns:
            df["num"] = 1
        frames.append(df.sort_values("logp", ascending=False).head(10))

    if not frames:
        print("Skipped GO BP/MF/CC fallback plot: no plottable GO terms.")
        return []

    df_plot = pd.concat(frames, ignore_index=True)
    df_plot = df_plot.sort_values(["ontology", "logp"], ascending=[True, False])
    labels = [
        f"{row.ontology}: {textwrap.fill(str(row.Term), width=42)}"
        for row in df_plot.itertuples(index=False)
    ]
    y_positions = list(range(len(df_plot)))

    fig_height = max(4.0, 0.45 * len(df_plot) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=300)
    size_values = df_plot["num"].astype(float).clip(lower=1) * 18
    scatter = ax.scatter(
        df_plot["fraction"],
        y_positions,
        c=df_plot["logp"],
        s=size_values,
        cmap=CMAP_CUSTOM,
        edgecolor="black",
        linewidth=0.25,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fractions of genes")
    ax.set_title("GO BP/MF/CC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(r"$-log_{10}$(FDR)")

    fig.tight_layout()
    return save_figure(fig, GO_MULTI_FIGURE_BASE)


def strip_go_id(term):
    return GO_ID_PATTERN.sub("", str(term)).strip()


def parse_gene_list(genes_text):
    if pd.isna(genes_text):
        return []
    return [gene.strip() for gene in str(genes_text).replace(",", ";").split(";") if gene.strip()]


def format_representative_genes(genes_text, requested_genes):
    if not requested_genes:
        return ""
    available = set(parse_gene_list(genes_text))
    genes = [gene for gene in requested_genes if gene in available]
    return ", ".join(genes)


def get_adjusted_pvalue(row):
    for column in ("Adjusted P-value", "Adjusted P Value", "FDR"):
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])
    raise ValueError("Enrichment result must contain an adjusted p-value column.")


def is_curated_go_candidate(term, fdr, overlap):
    term_lower = str(term).casefold()
    if fdr >= CURATED_GO_FDR_CUTOFF:
        return False
    if overlap < CURATED_GO_MIN_OVERLAP:
        return False
    return not any(pattern in term_lower for pattern in CURATED_GO_EXCLUDE_PATTERNS)


def curated_go_redundancy_key(term):
    term_lower = str(term).casefold()
    if "cholesterol homeostasis" in term_lower or "sterol homeostasis" in term_lower:
        return "sterol_homeostasis"
    if "lysosomal membrane" in term_lower or "lytic vacuole" in term_lower or term_lower == "lysosome":
        return "lysosomal_compartment"
    if "late endosome" in term_lower or "endosome membrane" in term_lower:
        return "endosomal_compartment"
    if term_lower == "microtubule" or "microtubule cytoskeleton" in term_lower:
        return "microtubule"
    if "apoptotic" in term_lower:
        return "apoptotic_process_regulation"
    if "negative regulation of protein localization" in term_lower:
        return "negative_regulation_of_protein_localization"
    return term_lower


def curated_go_preference_rank(term):
    term_lower = str(term).casefold()
    preferred_terms = {
        "cholesterol homeostasis": 0,
        "sterol homeostasis": 1,
        "lysosomal membrane": 0,
        "lytic vacuole membrane": 1,
        "lysosome": 2,
        "lytic vacuole": 3,
        "late endosome membrane": 0,
        "endosome membrane": 1,
        "late endosome": 2,
        "microtubule cytoskeleton": 0,
        "microtubule": 1,
        "negative regulation of apoptotic process": 0,
        "regulation of extrinsic apoptotic signaling pathway": 1,
        "negative regulation of extrinsic apoptotic signaling pathway": 1,
        "negative regulation of neuron apoptotic process": 2,
        "negative regulation of protein localization": 0,
        "negative regulation of protein localization to cell surface": 1,
    }
    return preferred_terms.get(term_lower, 0)


def curated_go_story_rank(name, term):
    priority_terms = CURATED_GO_PRIORITY_TERMS.get(name, ())
    priority_lookup = {priority_term: i for i, priority_term in enumerate(priority_terms)}
    return priority_lookup.get(term, len(priority_terms) + 100)


def build_curated_go_dataframe(name, result):
    if result is None or result.empty:
        return pd.DataFrame()
    if "Term" not in result.columns:
        raise ValueError(f"{name} enrichment result must contain a 'Term' column.")

    rows = []
    for _, row in result.iterrows():
        term_clean = strip_go_id(row["Term"])
        fdr = get_adjusted_pvalue(row)
        overlap = int(float(row.get("num", 0)))
        if not is_curated_go_candidate(term_clean, fdr, overlap):
            continue
        logfdr = -math.log10(max(fdr, 1e-300))
        genes = row.get("Genes", "")
        rows.append(
            {
                "ontology": name,
                "display_term": term_clean,
                "source_term": row["Term"],
                "fdr": fdr,
                "minus_log10_fdr": logfdr,
                "n": overlap,
                "genes": genes,
                "representative_genes": "",
                "redundancy_key": curated_go_redundancy_key(term_clean),
                "preference_rank": curated_go_preference_rank(term_clean),
                "story_rank": curated_go_story_rank(name, term_clean),
            }
        )

    if not rows:
        return pd.DataFrame()

    curated = pd.DataFrame(rows)
    curated = curated.sort_values(
        ["redundancy_key", "preference_rank", "fdr", "n"],
        ascending=[True, True, True, False],
    )
    curated = curated.drop_duplicates("redundancy_key", keep="first")
    curated = curated.sort_values(["story_rank", "fdr", "n"], ascending=[True, True, False]).head(
        CURATED_GO_MAX_TERMS.get(name, 8)
    )
    curated = curated.sort_values(["fdr", "n"], ascending=[True, False])
    curated = curated.reset_index(drop=True)
    curated["order"] = range(len(curated))
    return curated


def build_curated_go_summary_dataframe(results):
    frames = [
        build_curated_go_dataframe("GO_BP", results.get("GO_BP")),
        build_curated_go_dataframe("GO_CC", results.get("GO_CC")),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_curated_go_panel(ax, data, title, show_gene_labels=False):
    if data.empty:
        ax.axis("off")
        ax.set_title(title, fontsize=9)
        return

    data = data.sort_values("order", ascending=True).reset_index(drop=True)
    y_positions = list(range(len(data)))
    values = data["minus_log10_fdr"].astype(float)
    bars = ax.barh(y_positions, values, color=CURATED_GO_BAR_COLOR, height=0.72)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["display_term"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-log_{10}$(FDR)", fontsize=10)
    ax.set_title(title, fontsize=9, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="x", labelsize=9, length=3, width=0.8)
    ax.tick_params(axis="y", labelsize=9, length=0)

    x_max = max(values.max() * 1.18, 1.0)
    ax.set_xlim(0, x_max)
    text_offset = x_max * 0.015
    gene_x = x_max * 0.035

    for y_pos, bar, row in zip(y_positions, bars, data.itertuples(index=False)):
        value = float(row.minus_log10_fdr)
        ax.text(
            value + text_offset,
            y_pos,
            f"n={row.n}",
            va="center",
            ha="left",
            fontsize=6,
            color="black",
        )
        if show_gene_labels and row.representative_genes and value > x_max * 0.22:
            ax.text(
                gene_x,
                y_pos,
                row.representative_genes,
                va="center",
                ha="left",
                fontsize=8,
                color="white",
            )


def apply_curated_go_axis_width_compression(axes):
    axes = list(axes)
    for ax in axes:
        position = ax.get_position()
        ax.set_position(
            [
                position.x0,
                position.y0,
                position.width * CURATED_GO_AXIS_WIDTH_SCALE,
                position.height,
            ]
        )


def compact_curated_go_panel_gap(fig, axes):
    axes = list(axes)
    if len(axes) < 2:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_width = fig.get_size_inches()[0]
    target_gap = CURATED_GO_INTER_PANEL_GAP_INCHES / figure_width
    left_bbox = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    right_bbox = axes[1].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    excess_gap = right_bbox.x0 - left_bbox.x1 - target_gap
    if excess_gap <= 0:
        return

    right_position = axes[1].get_position()
    axes[1].set_position(
        [
            right_position.x0 - excess_gap,
            right_position.y0,
            right_position.width,
            right_position.height,
        ]
    )


def plot_curated_go_summary(results):
    curated = build_curated_go_summary_dataframe(results)
    if curated.empty:
        print("Skipped curated GO summary plot: no selected GO terms.")
        return []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = OUT_DIR / CURATED_GO_SOURCE_CSV
    curated.to_csv(source_path, index=False)
    print(f"Saved: {source_path}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 3.25),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.8},
    )
    plot_curated_go_panel(
        axes[0],
        curated[curated["ontology"] == "GO_BP"],
        CURATED_GO_PANEL_TITLES["GO_BP"],
        show_gene_labels=False,
    )
    plot_curated_go_panel(
        axes[1],
        curated[curated["ontology"] == "GO_CC"],
        CURATED_GO_PANEL_TITLES["GO_CC"],
        show_gene_labels=False,
    )
    fig.align_xlabels(axes)
    apply_curated_go_axis_width_compression(axes)
    compact_curated_go_panel_gap(fig, axes)
    return save_figure(fig, CURATED_GO_FIGURE_BASE)


def main():
    configure_matplotlib()
    validate_input_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    genes = load_candidate_genes()
    print(f"Candidate gene count: {len(genes)}")

    adata, background = load_adata_and_background_genes()
    missing_genes = find_missing_genes(adata, genes)
    print(f"{CELL_TYPE_FILTER} cell count: {adata.n_obs}")
    print(f"Missing gene count: {len(missing_genes)}")
    if missing_genes:
        raise ValueError(f"Missing genes in adata.var_names ({len(missing_genes)}): {missing_genes}")
    print(f"Background gene count: {len(background)}")
    print(f"Background min_cells: {BACKGROUND_MIN_CELLS}")

    results = {}
    for name in GENESET_CONFIGS:
        result = run_geneset_enrichment(name, genes, background)
        results[name] = result
        save_enrichment_result(name, result)

    plot_go_enrichment_figures(results)
    plot_curated_go_summary(results)
    plot_single_enrichment("KEGG", results["KEGG"])
    plot_single_enrichment("Reactome", results["Reactome"])

    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
