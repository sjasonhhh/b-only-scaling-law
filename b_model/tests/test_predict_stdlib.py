import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class StandardLibraryPredictionTests(unittest.TestCase):
    def test_prediction_runs_without_pipeline_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input.csv"
            output_path = temporary_path / "output.csv"
            input_path.write_text(
                "N_params_B,D_tokens_B,Q_score\n1.0,300,0.8\n10.0,0,1.0\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "b_model.predict",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
            )
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertGreater(float(rows[0]["prediction"]), 0.0)
            self.assertEqual(rows[1]["D_tokens_B_imputed"], "True")
            self.assertEqual(rows[1]["prediction_status"], "D_floor_imputed")

    def test_model_parameters_are_readable(self):
        model_path = ROOT / "b_model" / "results" / "B_only_model.json"
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertIn("base_model", payload)
        self.assertIn("quality_effect_model", payload)


if __name__ == "__main__":
    unittest.main()
