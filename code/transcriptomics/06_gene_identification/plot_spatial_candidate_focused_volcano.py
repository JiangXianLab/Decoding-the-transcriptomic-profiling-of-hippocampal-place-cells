from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


# ===== User-adjustable settings =====
ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
FIGSIZE = (5.0, 4.0)  # inches; keep equal for 1:1 output
TOP_N_UP_LABELS = 20
TOP_N_DOWN_LABELS = 20
POINT_SIZE = 31
LABEL_FONT_SIZE = 8
TITLE_FONT_SIZE = 11
AXIS_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LABEL_X_OFFSET = 0.035
LABEL_Y_OFFSET = 0.12
CONNECTOR_COLOR = "#6F6F6F"
CONNECTOR_WIDTH = 0.55

CANDIDATE_PATH = ROOT / "Data" / "results" / "spatial_candidate_final_genes_20260529.csv"
DEG_PATH = (
    ROOT
    / "Data"
    / "results"
    / "DEGs"
    / "FindMarkers_wilcoxon"
    / "place_nonplace_deg_wilcoxon_20260129.csv"
)
OUT_DIR = (
    ROOT
    / "Data"
    / "results"
    / "DEGs"
    / "FindMarkers_wilcoxon"
    / "Figures"
    / "PN"
)
OUT_BASE = OUT_DIR / "spatial_candidate_focused_volcano_20260529"

PALETTE = {"Up": "#E2A528", "Down": "#6FA3C6"}
TEXT_COLOR = "#222222"


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",  # keep text editable in Illustrator
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def load_plot_data():
    candidates = pd.read_csv(CANDIDATE_PATH)[["gene", "label"]].copy()
    candidates["gene"] = candidates["gene"].astype(str)
    candidates["label"] = candidates["label"].astype(str)

    pn = pd.read_csv(DEG_PATH, index_col=0)
    pn.index = pn.index.astype(str)

    plot_df = candidates.merge(
        pn[["p_val", "avg_log2FC"]],
        left_on="gene",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    assert len(plot_df) == 89, f"Expected 89 candidates, got {len(plot_df)}"
    missing = plot_df.loc[plot_df[["p_val", "avg_log2FC"]].isna().any(axis=1), "gene"].tolist()
    assert not missing, f"Missing PN DEG rows: {missing}"

    wrong_direction = plot_df.loc[
        ((plot_df["label"].eq("Up")) & (plot_df["avg_log2FC"] <= 0))
        | ((plot_df["label"].eq("Down")) & (plot_df["avg_log2FC"] >= 0)),
        ["gene", "label", "avg_log2FC"],
    ]
    assert wrong_direction.empty, wrong_direction.to_string(index=False)

    plot_df["p_val_clipped"] = pd.to_numeric(plot_df["p_val"], errors="coerce").clip(
        lower=1e-300, upper=1.0
    )
    plot_df["neglog10p"] = -np.log10(plot_df["p_val_clipped"])
    plot_df["score"] = plot_df["neglog10p"] * plot_df["avg_log2FC"].abs()
    return plot_df


def choose_label_genes(plot_df):
    label_genes = []
    label_genes.extend(
        plot_df.loc[plot_df["label"].eq("Up")].nlargest(TOP_N_UP_LABELS, "score")["gene"].tolist()
    )
    label_genes.extend(
        plot_df.loc[plot_df["label"].eq("Down")]
        .nlargest(TOP_N_DOWN_LABELS, "score")["gene"]
        .tolist()
    )
    return set(label_genes)


def plot_volcano(plot_df, label_set):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.90)

    for label in ["Up", "Down"]:
        sub = plot_df.loc[plot_df["label"].eq(label)]
        ax.scatter(
            sub["avg_log2FC"],
            sub["neglog10p"],
            s=POINT_SIZE,
            c=PALETTE[label],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            label=f"{label} ({len(sub)})",
            zorder=3,
        )

    ax.axvline(0, color="#8C8C8C", lw=0.8, ls="--", zorder=1)
    ax.axhline(-math.log10(0.05), color="#BDBDBD", lw=0.75, ls="--", zorder=1)

    texts = []
    label_points = []
    for _, row in plot_df.loc[plot_df["gene"].isin(label_set)].iterrows():
        is_up = row["avg_log2FC"] >= 0
        dx = LABEL_X_OFFSET if is_up else -LABEL_X_OFFSET
        ha = "left" if is_up else "right"
        texts.append(
            ax.text(
                row["avg_log2FC"] + dx,
                row["neglog10p"] + LABEL_Y_OFFSET,
                row["gene"],
                fontsize=LABEL_FONT_SIZE,
                color=TEXT_COLOR,
                ha=ha,
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.055",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                ),
                zorder=4,
            )
        )
        label_points.append((row["avg_log2FC"], row["neglog10p"]))

    xmin = min(plot_df["avg_log2FC"].min() - 0.17, -0.47)
    xmax = max(plot_df["avg_log2FC"].max() + 0.18, 0.58)
    ymax = plot_df["neglog10p"].max() + 0.62
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)

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

    # adjustText only draws arrows for labels it moves far enough. Draw an explicit
    # connector for every label so each gene name can be traced back to its point.
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

    ax.set_title("Spatial-selectivity-associated candidates, n = 89", fontsize=TITLE_FONT_SIZE, pad=7)
    ax.set_xlabel("log2FC (Place / active non-place)", fontsize=AXIS_FONT_SIZE)
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
    plot_df = load_plot_data()
    label_set = choose_label_genes(plot_df)
    fig = plot_volcano(plot_df, label_set)

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches=None)
    plt.close(fig)

    print(f"Saved: {OUT_BASE}.png/.svg/.pdf")
    print(f"Candidate count: {len(plot_df)}")
    print(f"Label counts: {plot_df['label'].value_counts().to_dict()}")
    print(
        "Labeled:",
        {
            "Up": int(plot_df[plot_df["label"].eq("Up")]["gene"].isin(label_set).sum()),
            "Down": int(plot_df[plot_df["label"].eq("Down")]["gene"].isin(label_set).sum()),
        },
    )


if __name__ == "__main__":
    main()
