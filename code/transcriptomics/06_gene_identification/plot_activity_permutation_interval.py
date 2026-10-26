import numpy as np
import pandas as pd
import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference

from plot_spatial_permutation_interval import (
    ADATA_PATH,
    DEG_DIR,
    N_PERM,
    OUT_DIR,
    RNG_SEED,
    TODAY,
    filter_deg,
    plot_interval,
)


ACTIVITY_RESULT_TAG = "PS_NS_noPC132"
PS_VALIDATION_SAMPLES = ("PC103", "PC106", "PC118")
NS_VALIDATION_SAMPLES = ("PC92", "PC105")
EXCLUDED_NS_SAMPLES = ("PC132",)
MIN_CONCORDANT_SAMPLES = 3


def get_activity_validation_contrasts():
    contrasts = []
    contrasts.extend(
        {
            "sample": sample,
            "comparison": "Place_vs_Silent",
            "active_condition": "place",
            "reference_condition": "silent",
        }
        for sample in PS_VALIDATION_SAMPLES
    )
    contrasts.extend(
        {
            "sample": sample,
            "comparison": "ActiveNonPlace_vs_Silent",
            "active_condition": "nonplace",
            "reference_condition": "silent",
        }
        for sample in NS_VALIDATION_SAMPLES
    )
    return contrasts


def build_activity_candidates():
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

    set_pn = deg_pn_up | deg_pn_down

    activity_up = (deg_ps_up & deg_ns_up) - set_pn
    activity_down = (deg_ps_down & deg_ns_down) - set_pn
    activity_candidate_df = pd.DataFrame(
        {
            "gene": sorted(activity_up) + sorted(activity_down),
            "label": ["Up"] * len(activity_up) + ["Down"] * len(activity_down),
        }
    )

    return activity_candidate_df.sort_values("gene").reset_index(drop=True)


def add_pseudobulk_replicates(adata, contrast):
    reps = []
    for condition in (contrast["active_condition"], contrast["reference_condition"]):
        mask = (adata.obs["sample"] == contrast["sample"]) & (adata.obs["condition"] == condition)
        if mask.sum() == 0:
            raise ValueError(
                f"No cells for sample={contrast['sample']}, condition={condition}"
            )
        subset = adata[mask].copy()
        subset.X = subset.layers["counts"]
        rep = sc.AnnData(X=subset.X.sum(axis=0, keepdims=True), var=subset.var)
        rep.obs_names = [f"{contrast['sample']}_{condition}"]
        rep.obs["sample"] = contrast["sample"]
        rep.obs["condition"] = condition
        rep.obs["active_condition"] = contrast["active_condition"]
        rep.obs["reference_condition"] = contrast["reference_condition"]
        rep.obs["comparison"] = contrast["comparison"]
        reps.append(rep)
    return reps


def build_sample_log2fc():
    adata = sc.read_h5ad(ADATA_PATH)
    adata = adata[adata.obs["cell_types"] == "CA1-ProS"].copy()
    contrasts = get_activity_validation_contrasts()
    validation_samples = [contrast["sample"] for contrast in contrasts]
    adata = adata[adata.obs["sample"].isin(validation_samples)].copy()

    pbs = []
    for contrast in contrasts:
        pbs.extend(add_pseudobulk_replicates(adata, contrast))

    pb = sc.concat(pbs)
    counts = pd.DataFrame(pb.X, index=pb.obs_names, columns=pb.var_names).astype(int)
    metadata = pb.obs[["sample", "condition", "active_condition", "reference_condition"]]
    assert (counts.index == metadata.index).all()

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
            ref_level=["condition", "silent"],
            inference=inference,
        )
        dds.fit_size_factors()
        norm = dds.layers["normed_counts"]
        cond = metadata_m["condition"].values
        active_condition = metadata_m["active_condition"].iloc[0]
        reference_condition = metadata_m["reference_condition"].iloc[0]
        idx_active = cond == active_condition
        idx_reference = cond == reference_condition
        log2fc = np.log2(
            (norm[idx_active, :].ravel() + 1) / (norm[idx_reference, :].ravel() + 1)
        )
        log2fc_dict[mouse] = pd.Series(log2fc, index=dds.var_names, name=mouse)

    return pd.concat(log2fc_dict, axis=1)


def permutation_result(activity_candidate_df, log2fc_df):
    label_s = activity_candidate_df.set_index("gene")["label"]
    up_genes = label_s.index[label_s.eq("Up")]
    down_genes = label_s.index[label_s.eq("Down")]

    consistent_up = log2fc_df.reindex(up_genes).gt(0).sum(axis=1).ge(MIN_CONCORDANT_SAMPLES)
    consistent_down = (
        log2fc_df.reindex(down_genes).lt(0).sum(axis=1).ge(MIN_CONCORDANT_SAMPLES)
    )
    target_genes = consistent_up.index[consistent_up].tolist() + consistent_down.index[
        consistent_down
    ].tolist()

    candidate_consistent_df = activity_candidate_df[activity_candidate_df["gene"].isin(target_genes)]
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
    result["supporting_samples"] = (
        log2fc_for_shuffle.gt(0).sum(axis=1).where(
            result.set_index("gene")["label"].eq("Up"),
            log2fc_for_shuffle.lt(0).sum(axis=1),
        )
    ).reindex(result["gene"]).to_numpy()
    return result


def save_final_candidates(result):
    final_df = (
        result.loc[result["pass_shuffle"], ["gene", "label"]]
        .sort_values("gene")
        .reset_index(drop=True)
    )
    out_path = OUT_DIR / f"activity_candidate_final_genes_{ACTIVITY_RESULT_TAG}_{TODAY}.csv"
    final_df.to_csv(out_path)
    print(f"Saved final activity candidates: {out_path} ({len(final_df)} genes)")
    return out_path, final_df


def select_label_genes(result):
    pass_df = result[result["pass_shuffle"]].copy()
    fail_df = result[~result["pass_shuffle"]].copy()
    labels = []
    labels.extend(pass_df.sort_values("obs_median_log2FC", ascending=False)["gene"].head(2))
    labels.extend(pass_df.sort_values("obs_median_log2FC", ascending=True)["gene"].head(1))
    labels.extend(fail_df.sort_values("abs_obs", ascending=False)["gene"].head(1))
    return list(dict.fromkeys(labels))


def main():
    activity_candidate_df = build_activity_candidates()
    activity_candidate_df.to_csv(
        OUT_DIR / f"activity_candidate_genes_directional_{ACTIVITY_RESULT_TAG}_{TODAY}.csv"
    )
    log2fc_df = build_sample_log2fc()
    result = permutation_result(activity_candidate_df, log2fc_df)
    result.to_csv(
        OUT_DIR / f"activity_permutation_interval_source_{ACTIVITY_RESULT_TAG}_{TODAY}.csv",
        index=False,
    )
    _, final_df = save_final_candidates(result)

    label_genes = select_label_genes(result)
    plot_interval(
        result,
        f"activity_permutation_interval_all_compact_{ACTIVITY_RESULT_TAG}_{TODAY}",
        top_n=None,
        show_gene_labels=False,
        label_genes=label_genes,
        height_override=4.2,
        width_override=2.65,
        marker_size=5,
        interval_lw=0.55,
        legend_loc="lower right",
    )
    plot_interval(
        result,
        f"activity_permutation_interval_top30_compact_{ACTIVITY_RESULT_TAG}_{TODAY}",
        top_n=30,
        show_gene_labels=True,
        height_override=5.0,
        width_override=3.25,
        marker_size=8,
        interval_lw=0.8,
        legend_loc="lower right",
    )
    print(
        f"Saved activity interval plots for {len(result)} concordant candidates; "
        f"{len(final_df)} passed shuffle. "
        f"Validation samples: {', '.join(log2fc_df.columns)}. "
        f"Excluded NS samples: {', '.join(EXCLUDED_NS_SAMPLES)}. "
        f"Labeled genes: {', '.join(label_genes)}"
    )


if __name__ == "__main__":
    main()
