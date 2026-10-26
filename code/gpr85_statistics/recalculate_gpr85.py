"""Reproduce table-based Gpr85 statistics for Fig. 4 and Extended Data Fig. 9.

Extracted from the 2026-10-20 manuscript audit, with the current panel mapping.
Image registration and FISH spot detection are upstream, separate workflows.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import multipletests


def run(source, output):
    d = pd.read_csv(source).sort_values(['mouse_id', 'in_vivo_suite2p_roi_id']).reset_index(drop=True)
    assert len(d) == 836 and d.is_place_cell.sum() == 435, 'This entry point targets the 836/435 manuscript dataset'
    assert set(d.mouse_id) == {'A81', 'B02', 'B10'}
    assert set(d.is_place_cell) == {0, 1}
    assert not d.duplicated(['mouse_id', 'in_vivo_suite2p_roi_id']).any()
    assert (d.cell_roi_area_um2 > 0).all()
    assert (d.gpr85_spot_count >= 0).all()
    assert np.array_equal(d.gpr85_positive_ge2_spots, (d.gpr85_spot_count >= 2).astype(int))
    density = d.gpr85_spot_count / d.cell_roi_area_um2 * 100
    relative = density / density.groupby(d.mouse_id).transform('mean')
    assert np.allclose(density, d.gpr85_spot_density_per_100_um2, rtol=1e-7, atol=1e-8)
    assert np.allclose(relative, d.gpr85_relative_spot_density, rtol=1e-7, atol=1e-8)

    specs = [('mean_significant_dff_all_frames', .01, 'Fig. 4i', 'Fig. 4j'),
             ('spatial_information_normalized', 1, 'ED Fig. 9c', 'ED Fig. 9g'),
             ('stability_lap_pairwise_r', .1, 'ED Fig. 9d', 'ED Fig. 9h'),
             ('mean_place_field_size_cm', 10, 'ED Fig. 9e', 'ED Fig. 9i'),
             ('place_field_number', 1, 'ED Fig. 9f', 'ED Fig. 9j')]
    rows, coefficients = [], []
    for prop, unit, pp, lp in specs:
        place_only = prop != 'mean_significant_dff_all_frames'
        sub = d.loc[d.is_place_cell.eq(1) if place_only else np.ones(len(d), bool)].dropna(subset=[prop])
        assert len(sub) == (435 if place_only else 836)
        x = sub[prop].to_numpy(float)
        y = sub.gpr85_positive_ge2_spots.to_numpy(float)
        assert np.isfinite(x).all()
        X = np.column_stack([np.ones(len(sub)), x, sub.mouse_id.eq('B02'), sub.mouse_id.eq('B10')]).astype(float)
        model = GLM(y, X, family=families.Binomial()).fit(tol=1e-12, maxiter=200)
        check = Logit(y, X).fit(disp=False, tol=1e-12, maxiter=200)
        assert model.converged and check.mle_retvals['converged']
        assert np.allclose(model.params, check.params, rtol=1e-8, atol=1e-10)
        mu = expit(X @ model.params)
        covariance = np.linalg.inv(X.T @ ((mu * (1 - mu))[:, None] * X))
        beta = model.params[1]
        se = np.sqrt(covariance[1, 1])
        p = 2 * stats.norm.sf(abs(beta / se))
        r, rp = stats.pearsonr(x, sub.gpr85_relative_spot_density)
        assert abs(r - np.corrcoef(x, sub.gpr85_relative_spot_density)[0, 1]) < 1e-12
        ci = np.exp(unit * (beta + np.array([-1, 1]) * stats.norm.ppf(.975) * se))
        rows.append(dict(property=prop, pearson_panel=pp, logistic_panel=lp, n=len(sub),
                         population='place' if place_only else 'all_registered', r=r, P_Pearson=rp,
                         beta=beta, se=se, OR_unit=unit, OR=np.exp(beta * unit),
                         CI_low=ci[0], CI_high=ci[1], P_Wald=p))
        for term, b, s in zip(['Intercept', prop, 'B02_vs_A81', 'B10_vs_A81'], model.params, model.bse):
            coefficients.append(dict(property=prop, term=term, beta=b, se=s))
    results = pd.DataFrame(rows)
    results['q_BH_5'] = multipletests(results.P_Wald, method='fdr_bh')[1]

    counts = []
    for mouse, s in d.groupby('mouse_id'):
        a = int(s.loc[s.is_place_cell.eq(1), 'gpr85_positive_ge2_spots'].sum())
        n_place = int(s.is_place_cell.sum())
        c = int(s.loc[s.is_place_cell.eq(0), 'gpr85_positive_ge2_spots'].sum())
        n_nonplace = len(s) - n_place
        counts.append(dict(mouse=mouse, place=n_place, nonplace=n_nonplace,
                           positive_place=a, positive_nonplace=c, fraction_place=a/n_place,
                           fraction_nonplace=c/n_nonplace,
                           log_OR=np.log(a*(n_nonplace-c)/((n_place-a)*c)),
                           variance=1/a+1/(n_place-a)+1/c+1/(n_nonplace-c)))
    ct = pd.DataFrame(counts)
    v, theta = ct.variance.to_numpy(), ct.log_OR.to_numpy()
    w = 1/v
    mean = np.sum(w*theta)/w.sum()
    Q = np.sum(w*(theta-mean)**2)
    tau = max(0, (Q-(len(ct)-1))/(w.sum()-(w*w).sum()/w.sum()))
    wr = 1/(v+tau)
    pooled = np.sum(wr*theta)/wr.sum()
    se = np.sqrt(1/wr.sum())
    ci = np.exp(pooled+np.array([-1, 1])*stats.norm.ppf(.975)*se)
    u = stats.mannwhitneyu(d.loc[d.is_place_cell.eq(1), 'gpr85_spot_density_per_100_um2'],
                          d.loc[d.is_place_cell.eq(0), 'gpr85_spot_density_per_100_um2'],
                          alternative='two-sided', method='asymptotic', use_continuity=True)
    group = dict(positive_fraction_OR=float(np.exp(pooled)), CI_low=float(ci[0]), CI_high=float(ci[1]),
                 P_Wald=float(2*stats.norm.sf(abs(pooled/se))), tau_squared=float(tau),
                 density_U=float(u.statistic), density_P=float(u.pvalue))

    # Animal-level sensitivity analysis is reported separately from primary cell-level statistics.
    means = d.groupby(['mouse_id', 'is_place_cell']).gpr85_spot_density_per_100_um2.mean().unstack()
    differences = (means[1]-means[0]).to_numpy()
    observed = float(differences.mean())
    null = np.array([np.mean(np.asarray(signs)*differences) for signs in itertools.product([-1,1], repeat=3)])
    sensitivity = dict(n_mice=3, n_sign_assignments=8, mean_place_minus_nonplace=observed,
                       per_mouse_differences=dict(zip(means.index, differences)),
                       exact_two_sided_sign_flip_P=float(np.mean(np.abs(null) >= abs(observed)-1e-12)))
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output/'association_statistics.csv', index=False)
    pd.DataFrame(coefficients).to_csv(output/'model_coefficients.csv', index=False)
    ct.to_csv(output/'per_mouse_counts.csv', index=False)
    (output/'group_statistics.json').write_text(json.dumps(group, indent=2)+'\n')
    (output/'mouse_density_sensitivity.json').write_text(json.dumps(sensitivity, indent=2)+'\n')
    provenance = dict(input_file=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      n_cells=len(d), n_place=int(d.is_place_cell.sum()), n_nonplace=int((1-d.is_place_cell).sum()),
                      python=platform.python_version(),
                      packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels']},
                      logistic_SE='ordinary cell-level maximum-likelihood SE; mouse identity fixed effects',
                      multiplicity='Benjamini-Hochberg adjustment across the five logistic Wald tests',
                      independent_GLM_Logit_and_Pearson_checks_passed=True)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(results.to_string(index=False))
    print(json.dumps(group))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
