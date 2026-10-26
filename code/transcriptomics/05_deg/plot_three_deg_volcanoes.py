from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from adjustText import adjust_text
except ImportError:  # labels will still be drawn, just without automatic repulsion
    adjust_text = None


# ===== User-adjustable settings =====
ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
DEG_DIR = ROOT / "Data" / "DEGs"
OUT_DIR = ROOT / "Data" / "results" / "DEGs" / "FindMarkers_wilcoxon" / "Figures" / "separate_volcanoes"
SOURCE_DATA_DIR = OUT_DIR / "source_data"
DATASET_TAG = "CA1ProsOnly_all_20260610"

P_COL = "p_val"
LOG2FC_COL = "avg_log2FC"
P_THRESHOLD = 0.05
FC_THRESHOLD = 0.1

FIGSIZE = (5.0, 4.0)
POINT_SIZE = 14
ALPHA_NS = 0.28
ALPHA_SIG = 0.72
TOP_N_LABELS_PER_SIDE = 12
LABEL_FONT_SIZE = 9
RASTERIZE_SCATTER_IN_VECTOR = True
VECTOR_RASTER_DPI = 600
PNG_DPI = 300
TIFF_DPI = 600
TITLE_FONT_SIZE = 11
AXIS_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LABEL_X_OFFSET = 0.035
LABEL_Y_OFFSET = 0.12
XLIM = (-1.0, 1.0)
YLIM = (0.0, 9.5)
XTICKS = np.arange(-1.0, 1.01, 0.5)
YTICKS = np.arange(0.0, 9.51, 2.0)

PALETTE = {"Up": "#E2A528", "Down": "#6FA3C6", "NS": "#D7D7D7"}
TEXT_COLOR = "#222222"
CONNECTOR_COLOR = "#6F6F6F"
CONNECTOR_WIDTH = 0.55

CONTRASTS = [
    {
        "key": "place_vs_active_nonplace",
        "file": "place_nonplace_deg_wilcoxon_CA1ProsOnly_all_20260610.csv",
        "title": "Place cell versus active non-place cell DEGs",
        "xlabel": "log2FC (Place / active non-place)",
    },
    {
        "key": "place_vs_silent",
        "file": "place_silent_deg_wilcoxon_CA1ProsOnly_all_20260610.csv",
        "title": "Place cell versus silent cell DEGs",
        "xlabel": "log2FC (Place / silent)",
    },
    {
        "key": "active_nonplace_vs_silent",
        "file": "nonplace_silent_deg_wilcoxon_CA1ProsOnly_all_20260610.csv",
        "title": "Active non-place cell versus silent cell DEGs",
        "xlabel": "log2FC (Active non-place / silent)",
    },
]


def safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": VECTOR_RASTER_DPI,
            "savefig.transparent": False,
            "savefig.bbox": "standard",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def load_and_classify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)

    required = {P_COL, LOG2FC_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df.copy()
    df[P_COL] = pd.to_numeric(df[P_COL], errors="coerce").fillna(1.0).clip(lower=1e-300, upper=1.0)
    df[LOG2FC_COL] = pd.to_numeric(df[LOG2FC_COL], errors="coerce")
    df = df.dropna(subset=[LOG2FC_COL])
    df["neglog10p"] = -np.log10(df[P_COL])
    df["sig"] = "NS"
    df.loc[(df[LOG2FC_COL] > FC_THRESHOLD) & (df[P_COL] <= P_THRESHOLD), "sig"] = "Up"
    df.loc[(df[LOG2FC_COL] < -FC_THRESHOLD) & (df[P_COL] <= P_THRESHOLD), "sig"] = "Down"
    df["score"] = df["neglog10p"] * df[LOG2FC_COL].abs()
    return df


def label_genes(ax: plt.Axes, df: pd.DataFrame):
    label_df = pd.concat(
        [
            df.loc[df["sig"].eq("Up")].nlargest(TOP_N_LABELS_PER_SIDE, "score"),
            df.loc[df["sig"].eq("Down")].nlargest(TOP_N_LABELS_PER_SIDE, "score"),
        ],
        axis=0,
    )
    if label_df.empty:
        return

    texts = []
    label_points = []
    for gene, row in label_df.iterrows():
        is_up = row[LOG2FC_COL] >= 0
        dx = LABEL_X_OFFSET if is_up else -LABEL_X_OFFSET
        ha = "left" if is_up else "right"
        texts.append(
            ax.text(
                row[LOG2FC_COL] + dx,
                row["neglog10p"] + LABEL_Y_OFFSET,
                gene,
                fontsize=LABEL_FONT_SIZE,
                color=TEXT_COLOR,
                ha=ha,
                va="center",
                zorder=5,
            )
        )
        label_points.append((row[LOG2FC_COL], row["neglog10p"]))

    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.08, 1.18),
            force_text=(0.48, 0.72),
            force_static=(0.20, 0.28),
            force_pull=(0.020, 0.030),
            ensure_inside_axes=True,
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
            arrowprops=None,
        )

    for text, (x, y) in zip(texts, label_points):
        tx, ty = text.get_position()
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(tx, ty),
            arrowprops=dict(
                arrowstyle="-",
                color=CONNECTOR_COLOR,
                lw=CONNECTOR_WIDTH,
                shrinkA=3,
                shrinkB=2,
                alpha=0.85,
            ),
            zorder=2,
        )


def plot_volcano(df: pd.DataFrame, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.90)

    for sig in ["NS", "Down", "Up"]:
        sub = df.loc[df["sig"].eq(sig)]
        ax.scatter(
            sub[LOG2FC_COL],
            sub["neglog10p"],
            s=POINT_SIZE if sig == "NS" else POINT_SIZE + 4,
            c=PALETTE[sig],
            edgecolors="none",
            alpha=ALPHA_NS if sig == "NS" else ALPHA_SIG,
            label=f"{sig} ({len(sub)})",
            rasterized=RASTERIZE_SCATTER_IN_VECTOR,
            zorder=1 if sig == "NS" else 3,
        )

    ax.axhline(-math.log10(P_THRESHOLD), ls="--", lw=0.8, color="#999999", zorder=0)
    ax.axvline(FC_THRESHOLD, ls="--", lw=0.8, color="#999999", zorder=0)
    ax.axvline(-FC_THRESHOLD, ls="--", lw=0.8, color="#999999", zorder=0)

    label_genes(ax, df)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=7)
    ax.set_xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("-log10(P value)", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, length=3.2)
    legend = ax.legend(
        title="Direction",
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        borderaxespad=0.0,
        handlelength=1.1,
        handletextpad=0.35,
    )
    for handle in legend.legend_handles:
        try:
            handle.set_sizes([38])
        except AttributeError:
            pass
    return fig


def main():
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for contrast in CONTRASTS:
        df = load_and_classify(DEG_DIR / contrast["file"])
        fig = plot_volcano(df, contrast["title"], contrast["xlabel"])
        out_base = OUT_DIR / (
            f"{contrast['key']}_volcano_{DATASET_TAG}_fc{safe_number(FC_THRESHOLD)}_p{safe_number(P_THRESHOLD)}"
        )
        source_data = df[[P_COL, LOG2FC_COL, "neglog10p", "sig", "score"]].copy()
        source_data.index.name = "gene"
        source_data.to_csv(
            SOURCE_DATA_DIR
            / (
                f"{contrast['key']}_volcano_{DATASET_TAG}_"
                f"fc{safe_number(FC_THRESHOLD)}_p{safe_number(P_THRESHOLD)}_source_data.csv"
            )
        )

        fig.savefig(out_base.with_suffix(".png"), dpi=PNG_DPI, bbox_inches=None)
        fig.savefig(out_base.with_suffix(".svg"), dpi=VECTOR_RASTER_DPI, bbox_inches=None)
        fig.savefig(out_base.with_suffix(".pdf"), dpi=VECTOR_RASTER_DPI, bbox_inches=None)
        fig.savefig(out_base.with_suffix(".tiff"), dpi=TIFF_DPI, bbox_inches=None)
        plt.close(fig)

        counts = df["sig"].value_counts().reindex(["Up", "Down", "NS"], fill_value=0).to_dict()
        summary.append((contrast["key"], out_base, counts))

    for key, out_base, counts in summary:
        print(f"{key}: {counts}")
        print(f"  saved: {out_base}.png/.svg/.pdf/.tiff")


if __name__ == "__main__":
    main()
