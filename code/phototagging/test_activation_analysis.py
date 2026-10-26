import tempfile
import unittest
from pathlib import Path

import pandas as pd

from activation_analysis import analyze_folder, discover_activation_files


class ActivationAnalysisTest(unittest.TestCase):
    def write_metrics(self, folder, name, activated):
        df = pd.DataFrame(
            {
                "roi_no": [1, 2, 3],
                "activated": activated,
                "post_pre_ratio": [1.2, 0.8, 1.5],
                "pre_mean": [10.0, 10.0, 10.0],
                "post_mean": [12.0, 8.0, 15.0],
                "in_bounds": [True, True, True],
            }
        )
        df.to_csv(folder / name, index=False)

    def test_discovers_four_channel_wavelength_files_and_applies_success_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_metrics(
                folder,
                "silent_cells_append_stimulatedsit_750_post_check_PAGFP_fold_roi_metrics.csv",
                [True, True, False],
            )
            self.write_metrics(
                folder,
                "silent_cells_append_stimulatedsit_750_post_check_PAmCherry_fold_roi_metrics.csv",
                [False, True, False],
            )
            self.write_metrics(
                folder,
                "nonplace_cells_append_stimulatedsit_830_post_check_PAGFP_fold_roi_metrics.csv",
                [True, True, False],
            )
            self.write_metrics(
                folder,
                "nonplace_cells_append_stimulatedsit_830_post_check_PAmCherry_fold_roi_metrics.csv",
                [True, False, True],
            )

            files = discover_activation_files(folder)
            self.assertEqual(files[750]["pagfp"].name, "silent_cells_append_stimulatedsit_750_post_check_PAGFP_fold_roi_metrics.csv")
            self.assertEqual(files[830]["pmcherry"].name, "nonplace_cells_append_stimulatedsit_830_post_check_PAmCherry_fold_roi_metrics.csv")

            results = analyze_folder(folder, export=False)
            self.assertEqual(int(results[750]["data"]["success"].sum()), 1)
            self.assertEqual(int(results[830]["data"]["success"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
