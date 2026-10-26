from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference


ROOT = Path(r"F:\Code\codex\PlaceCell_snRNAseq")
DEG_DIR = ROOT / "Data" / "DEGs"
ADATA_PATH = ROOT / "Data" / "FilteredData" / "All_QC_filtered_annotation_20260604.h5ad"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_COL = "p_val"
LFC_COL = "avg_log2FC"
P_TH = 0.05
LFC_TH = 0.1
N_PERM = 1000
RNG_SEED = 42
TODAY = date.today().strftime("%Y%m%d")

COL_UP = "#E2A528"
COL_DOWN = "#6FA3C6"
COL_PASS = "#E2A528"
COL_FAIL = "#6FA3C6"
COL_INTERVAL = "#B7B7B7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.size": 7,
    }
)


def filter_deg(df, direction="all"):
    if direction == "all":
        q = f"{P_COL} < @P_TH and abs({LFC_COL}) > @LFC_TH"
    elif direction == "up":
        q = f"{P_COL} < @P_TH and {LFC_COL} > @LFC_TH"
    elif direction == "down":
        q = f"{P_COL} < @P_TH and {LFC_COL} < -@LFC_TH"
    else:
        raise ValueError("direction must be one of {'all', 'up', 'down'}")
    return df.query(q).copy()


def build_spatial_candidates():
    df_ns = pd.read_csv(
        DEG_DIR / "SourceData_DEG_ActiveNonPlace_vs_Silent_Wilcoxon_CA1ProsOnly.csv",
        index_col=0,
    )
    df_ps = pd.read_csv(
        DEG_DIR / "SourceData_DEG_Place_vs_Silent_Wilcoxon_CA1ProsOnly.csv",
        index_col=0,
    )
    df_pn = pd.read_csv(
        DEG_DIR / "SourceData_DEG_Place_vs_ActiveNonPlace_Wilcoxon_CA1ProsOnly.csv",
        index_col=0,
    )

    deg_ps_up = set(filter_deg(df_ps, "up").index)
    deg_ps_down = set(filter_deg(df_ps, "down").index)
    deg_ns_up = set(filter_deg(df_ns, "up").index)
    deg_ns_down = set(filter_deg(df_ns, "down").index)
    deg_pn_up = set(filter_deg(df_pn, "up").index)
    deg_pn_down = set(filter_deg(df_pn, "down").index)

    set_ps = deg_ps_up | deg_ps_down
    set_ns = deg_ns_up | deg_ns_down
    set_pn = deg_pn_up | deg_pn_down

    spatial_candidate_genes = (set_pn & set_ps) - set_ns
    spatial_candidate_df = pd.DataFrame({"gene": sorted(spatial_candidate_genes)})

    up_set = set(deg_pn_up)
    down_set = set(deg_pn_down)
    spatial_candidate_df["label"] = np.select(
        [
            spatial_candidate_df["gene"].isin(up_set)
            & ~spatial_candidate_df["gene"].isin(down_set),
            spatial_candidate_df["gene"].isin(down_set)
            & ~spatial_candidate_df["gene"].isin(up_set),
            spatial_candidate_df["gene"].isin(up_set)
            & spatial_candidate_df["gene"].isin(down_set),
        ],
        ["Up", "Down", "Discordant"],
        default=pd.NA,
    )
    return spatial_candidate_df.dropna(subset=["label"])


def build_sample_log2fc():
    adata = sc.read_h5ad(ADATA_PATH)
    adata = adata[adata.obs["cell_types"] == "CA1-ProS"].copy()
    adata = adata[adata.obs["sample"].isin(["PC94", "PC110", "PC112", "PC114"])].copy()

    pbs = []
    for sample in adata.obs["sample"].unique():
        for condition in adata.obs["condition"].unique():
            mask = (adata.obs["sample"] == sample) & (adata.obs["condition"] == condition)
            if mask.sum() == 0:
                raise ValueError(f"No cells for sample={sample}, condition={condition}")
            subset = adata[mask].copy()
            subset.X = subset.layers["counts"]
            rep = sc.AnnData(X=subset.X.sum(axis=0, keepdims=True), var=subset.var)
            rep.obs_names = [f"{sample}_{condition}"]
            rep.obs["sample"] = sample
            rep.obs["condition"] = condition
            pbs.append(rep)

    pb = sc.concat(pbs)
    counts = pd.DataFrame(pb.X, index=pb.obs_names, columns=pb.var_names).astype(int)
    metadata = pb.obs[["sample", "condition"]]

    inference = DefaultInference(n_cpus=8)
    log2fc_dict = {}
    for mouse in metadata["sample"].unique():
        mask = metadata["sample"] == mouse
        counts_m = counts.loc[mask]
        metadata_m = metadata.loc[mask]
        if counts_m.shape[0] != 2:
            print(f"Skip {mouse}: {counts_m.shape[0]} samples")
            continue

        dds = DeseqDataSet(
            counts=counts_m,
            metadata=metadata_m,
            design_factors=["condition"],
            ref_level=["condition", "nonplace"],
            inference=inference,
        )
        dds.fit_size_factors()
        norm = dds.layers["normed_counts"]
        cond = metadata_m["condition"].values
        idx_place = cond == "place"
        idx_nonplace = cond == "nonplace"
        log2fc = np.log2((norm[idx_place, :].ravel() + 1) / (norm[idx_nonplace, :].ravel() + 1))
        log2fc_dict[mouse] = pd.Series(log2fc, index=dds.var_names, name=mouse)

    return pd.concat(log2fc_dict, axis=1)


def permutation_result(spatial_candidate_df, log2fc_df):
    label_s = spatial_candidate_df.set_index("gene")["label"]
    up_genes = label_s.index[label_s.eq("Up")]
    down_genes = label_s.index[label_s.eq("Down")]

    consistent_up = log2fc_df.reindex(up_genes).gt(0).sum(axis=1).ge(3)
    consistent_down = log2fc_df.reindex(down_genes).lt(0).sum(axis=1).ge(3)
    target_genes = consistent_up.index[consistent_up].tolist() + consistent_down.index[
        consistent_down
    ].tolist()

    candidate_consistent_df = spatial_candidate_df[spatial_candidate_df["gene"].isin(target_genes)]
    log2fc_for_shuffle = log2fc_df.reindex(candidate_consistent_df["gene"]).dropna(how="all")
    obs_stat = log2fc_for_shuffle.median(axis=1)

    rng = np.random.default_rng(RNG_SEED)
    x = log2fc_for_shuffle.to_numpy()
    _, n_samples = x.shape
    signs = rng.choice([-1, 1], size=(N_PERM, n_samples))
    perm_stats = np.median(x[None, :, :] * signs[:, None, :], axis=2)

    thr_lo = np.percentile(perm_stats, 5, axis=0)
    thr_hi = np.percentile(perm_stats, 95, axis=0)
    obs = obs_stat.to_numpy()
    pass_shuffle = (obs <= thr_lo) | (obs >= thr_hi)

    result = pd.DataFrame(
        {
            "gene": obs_stat.index,
            "obs_median_log2FC": obs_stat.values,
            "perm_thr_lo": thr_lo,
            "perm_thr_hi": thr_hi,
            "pass_shuffle": pass_shuffle,
        }
    )
    result = result.merge(candidate_consistent_df, on="gene", how="left")
    result["abs_obs"] = result["obs_median_log2FC"].abs()
    result["exceed_margin"] = np.where(
        result["obs_median_log2FC"] >= result["perm_thr_hi"],
        result["obs_median_log2FC"] - result["perm_thr_hi"],
        np.where(
            result["obs_median_log2FC"] <= result["perm_thr_lo"],
            result["perm_thr_lo"] - result["obs_median_log2FC"],
            0.0,
        ),
    )
    return result


def plot_interval(
    df,
    out_stem,
    top_n=None,
    show_gene_labels=True,
    label_genes=None,
    height_override=None,
    width_override=None,
    marker_size=14,
    interval_lw=1.0,
    legend_loc="lower right",
):
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.sort_values("abs_obs", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("obs_median_log2FC")

    n = len(plot_df)
    height = height_override if height_override is not None else max(3.0, min(10.0, 0.16 * n + 1.2))
    width = width_override if width_override is not None else (3.7 if show_gene_labels else 3.2)
    fig, ax = plt.subplots(figsize=(width, height))

    y = np.arange(n)
    colors = np.where(plot_df["pass_shuffle"], COL_PASS, COL_FAIL)
    ax.hlines(
        y,
        plot_df["perm_thr_lo"],
        plot_df["perm_thr_hi"],
        color=COL_INTERVAL,
        linewidth=interval_lw,
        zorder=1,
    )
    ax.scatter(plot_df["obs_median_log2FC"], y, c=colors, s=marker_size, linewidths=0, zorder=3)
    ax.axvline(0, color="#5F5F5F", linewidth=0.8, linestyle="--", zorder=0)

    if label_genes:
        label_set = set(label_genes)
        for yi, row in enumerate(plot_df.itertuples(index=False)):
            if row.gene not in label_set:
                continue
            xval = row.obs_median_log2FC
            ha = "left" if xval >= 0 else "right"
            xoff = 4 if xval >= 0 else -4
            ax.annotate(
                row.gene,
                xy=(xval, yi),
                xytext=(xoff, 0),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=5.2,
                color="#202020",
                arrowprops=dict(arrowstyle="-", lw=0.35, color="#6F6F6F", shrinkA=0, shrinkB=2),
                zorder=4,
            )

    if show_gene_labels:
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["gene"], fontsize=5.4)
    else:
        ax.set_yticks([])
        ax.set_ylabel("Candidate genes")

    ax.set_xlabel("Median sample-level log2FC")
    ax.set_title("Permutation-based robustness", fontsize=8, pad=5)

    pass_n = int(plot_df["pass_shuffle"].sum())
    fail_n = int((~plot_df["pass_shuffle"]).sum())
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_PASS, markeredgecolor="none", markersize=4),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_FAIL, markeredgecolor="none", markersize=4),
        plt.Line2D([0], [0], color=COL_INTERVAL, linewidth=1.2),
    ]
    labels = [f"Pass ({pass_n})", f"Fail ({fail_n})", "5-95% shuffle interval"]
    ax.legend(handles, labels, frameon=False, loc=legend_loc, fontsize=5.8, handlelength=1.5)
    ax.tick_params(axis="x", labelsize=6.5, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else None
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    spatial_candidate_df = build_spatial_candidates()
    log2fc_df = build_sample_log2fc()
    result = permutation_result(spatial_candidate_df, log2fc_df)
    result.to_csv(OUT_DIR / f"spatial_permutation_interval_source_{TODAY}.csv", index=False)

    plot_interval(
        result,
        f"spatial_permutation_interval_top40_{TODAY}",
        top_n=40,
        show_gene_labels=True,
    )
    plot_interval(
        result,
        f"spatial_permutation_interval_all_{TODAY}",
        top_n=None,
        show_gene_labels=False,
    )
    plot_interval(
        result,
        f"spatial_permutation_interval_all_compact_{TODAY}",
        top_n=None,
        show_gene_labels=False,
        label_genes=["Trim7", "Fabp3", "Vim", "Zcchc9-ps"],
        height_override=4.2,
        width_override=2.65,
        marker_size=5,
        interval_lw=0.55,
        legend_loc="lower right",
    )
    plot_interval(
        result,
        f"spatial_permutation_interval_top30_compact_{TODAY}",
        top_n=30,
        show_gene_labels=True,
        height_override=5.0,
        width_override=3.25,
        marker_size=8,
        interval_lw=0.8,
        legend_loc="lower right",
    )
    print(
        f"Saved interval plots for {len(result)} concordant candidates; "
        f"{int(result['pass_shuffle'].sum())} passed shuffle."
    )


if __name__ == "__main__":
    main()
