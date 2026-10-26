import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import glob
import re
import os

data_dir = r'F:\Code\opencode\Histogram\PAGFP'
files = sorted(glob.glob(os.path.join(data_dir, '*_post_check_fold_roi_metrics.csv')))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.weight'] = 'normal'

x_min, x_max = 0.5, 9
num_bins = 40
bin_width = (x_max - x_min) / num_bins

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

colors = ['#6FA3C6', '#E2A528']
data_dict = {}

for i, filepath in enumerate(files):
    match = re.search(r'_(\d{3})_', os.path.basename(filepath))
    if not match:
        continue
    label = match.group(1)

    df = pd.read_csv(filepath)
    valid_data = df['post_pre_ratio'].dropna()
    valid_data = valid_data[(valid_data >= x_min) & (valid_data <= x_max)]
    data_dict[label] = valid_data

    ax = axes[i]
    kde = gaussian_kde(valid_data)
    x_kde = np.linspace(x_min, x_max, 200)
    y_kde = kde(x_kde) * bin_width
    ax.plot(x_kde, y_kde, color=colors[i % len(colors)], linewidth=2.5)

    median_val = valid_data.median()
    ax.axvline(median_val, color=colors[i % len(colors)], linestyle='--', linewidth=1.5)

    ax.set_title(label, fontsize=13)
    ax.set_xlabel('fold change', fontsize=12)
    ax.set_ylabel('Fraction of cells', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print(f'{label}: N={len(valid_data)}, Median={median_val:.2f}')

ax_comb = axes[2]
for i, (label, valid_data) in enumerate(data_dict.items()):
    kde = gaussian_kde(valid_data)
    x_kde = np.linspace(x_min, x_max, 200)
    y_kde = kde(x_kde) * bin_width
    ax_comb.plot(x_kde, y_kde, color=colors[i % len(colors)], linewidth=2.5, label=label)

    median_val = valid_data.median()
    ax_comb.axvline(median_val, color=colors[i % len(colors)], linestyle='--', linewidth=1.5)

ax_comb.set_title('Combined', fontsize=13)
ax_comb.set_xlabel('fold change', fontsize=12)
ax_comb.set_ylabel('Fraction of cells', fontsize=12)
ax_comb.set_xlim(x_min, x_max)
ax_comb.spines['top'].set_visible(False)
ax_comb.spines['right'].set_visible(False)
ax_comb.legend(fontsize=9, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig(r'F:\Code\opencode\Histogram\post_pre_ratio_histogram.png', bbox_inches='tight')
plt.savefig(r'F:\Code\opencode\Histogram\post_pre_ratio_histogram.svg', bbox_inches='tight')
print('Figure saved.')
