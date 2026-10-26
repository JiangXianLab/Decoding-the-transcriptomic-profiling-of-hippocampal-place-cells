from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# The processed per-mouse metric tables are kept with Figure 2 data, not with
# the plotting code.
ROOT_DIR = Path(__file__).resolve().parents[2] / "03_Data" / "ProcessedMetrics"
OUTPUT_DIR_NAME = "ecdf_merged"
OUTPUT_STEM = "post_pre_ratio_ecdf"
X_MIN, X_MAX = 0.5, 9
METRIC_COLUMN = "post_pre_ratio"
WAVELENGTHS = (750, 830)
CHANNEL_LABELS = {
    "PAGFP": "PAGFP",
    "PAmCherry": "PAmCherry",
}
GROUP_ORDER = ("750_PAGFP", "750_PAmCherry", "830_PAGFP", "830_PAmCherry")
COLORS = {
    "750": "#6FA3C6",
    "830": "#E2A528",
}
PLOT_PANELS = (
    ("PAGFP", "PAGFP fold change"),
    ("PAmCherry", "PAmCherry fold change"),
)


def configure_matplotlib():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.weight"] = "normal"


def detect_wavelength(path: Path) -> int | None:
    text = path.name.lower()
    for wavelength in WAVELENGTHS:
        if re.search(rf"(^|[^0-9]){wavelength}([^0-9]|$)", text):
            return wavelength
    return None


def detect_channel(path: Path) -> str | None:
    text = path.name.lower()
    if "pagfp" in text:
        return "PAGFP"
    if "pamcherry" in text or "pmcherry" in text:
        return "PAmCherry"
    return None


def iter_pc_folders(root_dir: Path):
    for folder in sorted(root_dir.iterdir()):
        if folder.is_dir() and re.fullmatch(r"PC\d+", folder.name):
            yield folder


def build_merged_tables(root_dir: Path | str) -> dict[str, pd.DataFrame]:
    root_dir = Path(root_dir)
    tables: dict[str, list[pd.DataFrame]] = {group: [] for group in GROUP_ORDER}

    for mouse_dir in iter_pc_folders(root_dir):
        seen = set()
        for filepath in sorted(mouse_dir.glob("*_roi_metrics.csv")):
            wavelength = detect_wavelength(filepath)
            channel = detect_channel(filepath)
            if wavelength is None or channel is None:
                continue

            group = f"{wavelength}_{channel}"
            if group not in tables:
                continue
            if group in seen:
                raise ValueError(f"{mouse_dir.name} 中 {group} 匹配到多个文件")
            seen.add(group)

            df = pd.read_csv(filepath)
            df.insert(0, "source_file", filepath.name)
            df.insert(0, "mouse", mouse_dir.name)
            df.insert(0, "channel", channel)
            df.insert(0, "wavelength", wavelength)
            tables[group].append(df)

    merged = {}
    for group, frames in tables.items():
        if not frames:
            raise ValueError(f"没有找到 {group} 对应的 CSV 文件")
        merged[group] = pd.concat(frames, ignore_index=True)
    return merged


def save_merged_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for group in GROUP_ORDER:
        path = output_dir / f"{group}.csv"
        tables[group].to_csv(path, index=False)
        paths[group] = path
    return paths


def load_groups_from_tables(tables: dict[str, pd.DataFrame], x_min: float, x_max: float):
    groups = []
    for group in GROUP_ORDER:
        df = tables[group]
        if METRIC_COLUMN not in df.columns:
            raise ValueError(f"{group} 缺少列 {METRIC_COLUMN!r}")

        data = pd.to_numeric(df[METRIC_COLUMN], errors="coerce").dropna()
        data = data[(data >= x_min) & (data <= x_max)]
        if len(data) < 2 or data.nunique() < 2:
            raise ValueError(f"{group} 过滤后可绘制数据不足")

        groups.append(
            {
                "label": group,
                "data": data,
                "median": data.median(),
                "n": len(data),
            }
        )
    return groups


def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_groups(groups, output_dir: Path, x_min: float, x_max: float):
    groups_by_label = {group["label"]: group for group in groups}
    fig, axes = plt.subplots(2, 1, figsize=(3.0, 5.2), sharex=False)

    for ax, (channel, xlabel) in zip(axes, PLOT_PANELS):
        for wavelength in ("750", "830"):
            group_label = f"{wavelength}_{channel}"
            group = groups_by_label[group_label]
            x_ecdf, y_ecdf = ecdf(group["data"])
            ax.plot(x_ecdf, y_ecdf, color=COLORS[wavelength], linewidth=1.5, label=wavelength)
            print(f'{group["label"]}: N={group["n"]}, Median={group["median"]:.2f}')

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Cumulative fraction", fontsize=14)
        ax.tick_params(labelsize=12, length=5, width=1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        fontsize=12,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.22),
        ncol=2,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    plt.tight_layout(h_pad=1.0)
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    svg_path = output_dir / f"{OUTPUT_STEM}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return fig, axes, (png_path, svg_path)


def parse_args():
    parser = argparse.ArgumentParser(description="合并所有 PC 文件夹的 4 类 CSV 并绘制 post_pre_ratio ECDF")
    parser.add_argument("root_dir", nargs="?", default=ROOT_DIR, help="包含 PC* 子文件夹的目录")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 root_dir/ecdf_merged")
    parser.add_argument("--x-min", type=float, default=X_MIN)
    parser.add_argument("--x-max", type=float, default=X_MAX)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_dir / OUTPUT_DIR_NAME

    configure_matplotlib()
    tables = build_merged_tables(root_dir)
    csv_paths = save_merged_tables(tables, output_dir)
    groups = load_groups_from_tables(tables, args.x_min, args.x_max)
    _, _, figure_paths = plot_groups(groups, output_dir, args.x_min, args.x_max)

    print("Merged CSV saved:")
    for group in GROUP_ORDER:
        print(f"  {csv_paths[group]}")
    print("Figure saved:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
