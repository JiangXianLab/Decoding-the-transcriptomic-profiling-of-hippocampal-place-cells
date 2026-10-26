import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_ecdf import build_merged_tables, load_groups_from_tables, plot_groups


class PlotEcdfTest(unittest.TestCase):
    def write_metrics(self, folder, name, values):
        df = pd.DataFrame(
            {
                "roi_no": range(len(values)),
                "post_pre_ratio": values,
                "activated": [True] * len(values),
            }
        )
        df.to_csv(folder / name, index=False)

    def test_builds_four_merged_tables_from_pc_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc92 = root / "PC92"
            pc94 = root / "PC94"
            pc92.mkdir()
            pc94.mkdir()

            for folder in (pc92, pc94):
                self.write_metrics(folder, "cells_750_post_check_PAGFP_fold_roi_metrics.csv", [1.0, 2.0])
                self.write_metrics(folder, "cells_750_post_check_PAmCherry_fold_roi_metrics.csv", [3.0])
                self.write_metrics(folder, "cells_830_post_check_PAGFP_fold_roi_metrics.csv", [4.0])
                self.write_metrics(folder, "cells_830_post_check_PAmCherry_fold_roi_metrics.csv", [5.0, 6.0])

            tables = build_merged_tables(root)

            self.assertEqual(len(tables["750_PAGFP"]), 4)
            self.assertEqual(len(tables["750_PAmCherry"]), 2)
            self.assertEqual(len(tables["830_PAGFP"]), 2)
            self.assertEqual(len(tables["830_PAmCherry"]), 4)
            self.assertEqual(set(tables["750_PAGFP"]["mouse"]), {"PC92", "PC94"})

    def test_plots_pagfp_and_pamcherry_as_separate_panels(self):
        tables = {
            "750_PAGFP": pd.DataFrame({"post_pre_ratio": [1.0, 2.0, 3.0]}),
            "830_PAGFP": pd.DataFrame({"post_pre_ratio": [1.5, 2.5, 3.5]}),
            "750_PAmCherry": pd.DataFrame({"post_pre_ratio": [0.8, 1.0, 1.2]}),
            "830_PAmCherry": pd.DataFrame({"post_pre_ratio": [2.0, 3.0, 4.0]}),
        }

        with tempfile.TemporaryDirectory() as tmp:
            groups = load_groups_from_tables(tables, 0.5, 9)
            fig, axes, _ = plot_groups(groups, Path(tmp), 0.5, 9)

            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[0].get_xlabel(), "PAGFP fold change")
            self.assertEqual(axes[1].get_xlabel(), "PAmCherry fold change")
            self.assertEqual([line.get_label() for line in axes[0].lines], ["750", "830"])
            self.assertEqual([line.get_label() for line in axes[1].lines], ["750", "830"])
            fig.clf()


if __name__ == "__main__":
    unittest.main()
