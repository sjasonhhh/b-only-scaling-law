import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from b_model.pipeline import D_FLOOR, base_features, metric_dict, model_from_dict, quality_effect_stability, quality_features, spearman_correlation


class PipelineUnitTests(unittest.TestCase):
    def test_metrics_perfect_prediction(self):
        values = [1.0, 2.0, 3.0]
        metrics = metric_dict(values, values)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_feature_shapes(self):
        frame = pd.DataFrame(
            {
                "N_params_B": [0.07, 1.0],
                "D_tokens_B": [10.0, 300.0],
                "Q_score": [0.1, 1.0],
            }
        )
        self.assertEqual(base_features(frame).shape, (2, 8))
        self.assertEqual(quality_features(frame).shape, (2, 15))

    def test_spearman_reverses(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)

    def test_features_are_finite(self):
        frame = pd.DataFrame(
            {
                "N_params_B": [0.07],
                "D_tokens_B": [10.0],
                "Q_score": [0.1],
            }
        )
        self.assertTrue(np.isfinite(base_features(frame)).all())
        self.assertTrue(np.isfinite(quality_features(frame)).all())

    def test_nonpositive_data_uses_finite_floor(self):
        frame = pd.DataFrame(
            {
                "N_params_B": [100.0],
                "D_tokens_B": [0.0],
                "Q_score": [1.0],
            }
        )
        self.assertTrue(np.isfinite(base_features(frame)).all())
        self.assertGreater(D_FLOOR, 0.0)

    def test_serialized_model_has_two_components(self):
        model_path = Path("b_model/results/B_only_model.json")
        if model_path.exists():
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            base_model, quality_model = model_from_dict(payload)
            self.assertIsNotNone(base_model.coefficients)
            self.assertIsNotNone(quality_model.coefficients)

    def test_quality_effect_stability_diagnostic(self):
        model_path = Path("b_model/results/B_only_model.json")
        if model_path.exists():
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            base_model, quality_model = model_from_dict(payload)
            quality_values = np.linspace(0.0, 1.0, 6)
            frame = pd.DataFrame(
                {
                    "N_params_B": [0.07] * 6 + [1.0] * 6,
                    "D_tokens_B": [10.0] * 6 + [300.0] * 6,
                    "Q_score": list(quality_values) * 2,
                }
            )
            diagnostic = quality_effect_stability(base_model, quality_model, frame)
            self.assertGreaterEqual(diagnostic["direction_consistency_rate"], 0.95)


if __name__ == "__main__":
    unittest.main()
