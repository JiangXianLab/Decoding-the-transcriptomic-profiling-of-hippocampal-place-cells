"""激活成功率分析脚本

用法:
    python activation_analysis.py PC92
    python activation_analysis.py "F:/Figures/PlaceCells/Figure2/Data/Statistics/PC92"

目标文件夹中需要能找到 4 个 roi_metrics CSV:
    750 + PAGFP
    750 + PAmCherry
    830 + PAGFP
    830 + PAmCherry

750波段: PAGFP 激活，PAmCherry 不激活
830波段: PAGFP 和 PAmCherry 都激活
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["roi_no", "activated", "post_pre_ratio", "pre_mean", "post_mean", "in_bounds"]
WAVELENGTHS = (750, 830)
CHANNELS = ("pagfp", "pmcherry")


def _detect_wavelength(path: Path) -> int | None:
    text = str(path).lower()
    for wavelength in WAVELENGTHS:
        if re.search(rf"(^|[^0-9]){wavelength}([^0-9]|$)", text):
            return wavelength
    return None


def _detect_channel(path: Path) -> str | None:
    text = str(path).lower()
    if "pagfp" in text:
        return "pagfp"
    if "pamcherry" in text or "pmcherry" in text:
        return "pmcherry"
    return None


def discover_activation_files(folder: Path | str) -> dict[int, dict[str, Path]]:
    """Find the 750/830 PAGFP/PAmCherry CSV files under folder."""
    folder = Path(folder)
    candidates = sorted(folder.rglob("*_roi_metrics.csv"))
    matches: dict[int, dict[str, list[Path]]] = {
        wavelength: {channel: [] for channel in CHANNELS} for wavelength in WAVELENGTHS
    }

    for path in candidates:
        wavelength = _detect_wavelength(path)
        channel = _detect_channel(path)
        if wavelength is not None and channel is not None:
            matches[wavelength][channel].append(path)

    discovered: dict[int, dict[str, Path]] = {}
    problems = []
    for wavelength in WAVELENGTHS:
        discovered[wavelength] = {}
        for channel in CHANNELS:
            paths = matches[wavelength][channel]
            if len(paths) == 1:
                discovered[wavelength][channel] = paths[0]
            elif not paths:
                problems.append(f"缺少 {wavelength} {channel} 文件")
            else:
                joined = "\n    ".join(str(path) for path in paths)
                problems.append(f"{wavelength} {channel} 匹配到多个文件:\n    {joined}")

    if problems:
        raise FileNotFoundError(
            f"无法在 {folder} 中唯一确定 4 个输入文件:\n" + "\n".join(f"- {problem}" for problem in problems)
        )

    return discovered


def load_pair(files: dict[str, Path]) -> pd.DataFrame:
    pagfp = pd.read_csv(files["pagfp"])
    pmcherry = pd.read_csv(files["pmcherry"])

    for channel, df in (("PAGFP", pagfp), ("PAmCherry", pmcherry)):
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"{channel} 文件缺少列: {missing}")

    return pagfp[REQUIRED_COLS].merge(
        pmcherry[REQUIRED_COLS],
        on="roi_no",
        suffixes=("_pagfp", "_pmcherry"),
    )


def calculate_success(wavelength: int, data: pd.DataFrame) -> pd.Series:
    if wavelength == 750:
        return data["activated_pagfp"] & (~data["activated_pmcherry"])
    if wavelength == 830:
        return data["activated_pagfp"] & data["activated_pmcherry"]
    raise ValueError(f"不支持的波段: {wavelength}")


def analyze_folder(folder: Path | str, export: bool = True) -> dict[int, dict[str, object]]:
    folder = Path(folder)
    files = discover_activation_files(folder)
    results: dict[int, dict[str, object]] = {}

    for wavelength in WAVELENGTHS:
        merged = load_pair(files[wavelength])
        merged["success"] = calculate_success(wavelength, merged)
        if export:
            merged.to_csv(folder / f"result_{wavelength}_activation.csv", index=False)

        results[wavelength] = {
            "files": files[wavelength],
            "data": merged,
            "success_count": int(merged["success"].sum()),
            "total_count": int(len(merged)),
            "success_rate": float(merged["success"].mean()) if len(merged) else 0.0,
        }

    return results


def print_summary(results: dict[int, dict[str, object]]) -> None:
    for wavelength in WAVELENGTHS:
        result = results[wavelength]
        data = result["data"]
        files = result["files"]
        success_rule = "PAGFP激活, PAmCherry不激活" if wavelength == 750 else "PAGFP和PAmCherry都激活"

        print("=" * 60)
        print(f"{wavelength}波段: {success_rule}")
        print("=" * 60)
        print(f"PAGFP文件: {files['pagfp']}")
        print(f"PAmCherry文件: {files['pmcherry']}")
        print(f"总ROI数: {result['total_count']}")
        print(f"激活成功数: {result['success_count']}")
        print(f"成功率: {result['success_count']}/{result['total_count']} = {result['success_rate']:.2%}")
        print()
        print("各组合统计 (PAGFP, PAmCherry):")
        combo = data.groupby(["activated_pagfp", "activated_pmcherry"]).size()
        for (pagfp_active, pmcherry_active), count in combo.items():
            success = (pagfp_active and not pmcherry_active) if wavelength == 750 else (pagfp_active and pmcherry_active)
            label = "成功" if success else "失败"
            print(f"  PAGFP={pagfp_active}, PAmCherry={pmcherry_active}: {count} ({label})")

        print()
        print(f"{wavelength}波段 - 失败案例 (共{int((~data['success']).sum())}个):")
        fail = data[~data["success"]].sort_values("roi_no")
        both_false = fail[~fail["activated_pagfp"] & ~fail["activated_pmcherry"]]
        pagfp_only = fail[fail["activated_pagfp"] & ~fail["activated_pmcherry"]]
        pmcherry_only = fail[~fail["activated_pagfp"] & fail["activated_pmcherry"]]
        both_true = fail[fail["activated_pagfp"] & fail["activated_pmcherry"]]
        print(f"  PAGFP和PAmCherry都未激活: {len(both_false)} 个 - ROI: {list(both_false['roi_no'])}")
        print(f"  仅PAGFP激活:             {len(pagfp_only)} 个 - ROI: {list(pagfp_only['roi_no'])}")
        print(f"  仅PAmCherry激活:         {len(pmcherry_only)} 个 - ROI: {list(pmcherry_only['roi_no'])}")
        print(f"  PAGFP和PAmCherry都激活:  {len(both_true)} 个 - ROI: {list(both_true['roi_no'])}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动计算指定文件夹中的 750/830 激活成功率")
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="包含 4 个 roi_metrics CSV 的文件夹，例如 PC92",
    )
    parser.add_argument("--no-export", action="store_true", help="只打印结果，不导出 result_*_activation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    results = analyze_folder(folder, export=not args.no_export)
    print_summary(results)
    if not args.no_export:
        print("已导出:")
        for wavelength in WAVELENGTHS:
            print(f"  {folder / f'result_{wavelength}_activation.csv'}")


if __name__ == "__main__":
    main()
