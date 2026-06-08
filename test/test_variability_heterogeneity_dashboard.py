import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess.utils.variability_heterogeneity_dashboard import (  # noqa: E402
    combine_variability_score,
)


class VariabilityCompositeScoreTests(unittest.TestCase):
    def test_composite_score_drops_all_nan_rows_without_runtime_warning(self):
        results = {
            "RI": {
                "MED_seg_medbeat": [1.0, 1.0, 1.0],
                "STD_seg_medbeat": [0.25, np.nan, 0.50],
                "MAD_seg_medbeat": [0.75, np.nan, np.nan],
            }
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            score = combine_variability_score(
                results,
                "RI",
                higher_metrics=["STD_seg_medbeat", "MAD_seg_medbeat"],
            )

        runtime_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual([], runtime_warnings)
        np.testing.assert_allclose(score, np.asarray([0.50, 0.50]))


if __name__ == "__main__":
    unittest.main()
