import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "predict.py"
P_COLUMNS = [f"p_{index}" for index in range(1, 18)]
INPUT_COLUMNS = ["N_params_B", "D_tokens_B", "Q_score", *P_COLUMNS]


def _input_row(parameter_count="1.0", token_count="300", quality="0.8"):
    return [parameter_count, token_count, quality, "1", *(["0"] * 16)]


class PredictionTests(unittest.TestCase):
    def _run(self, rows, columns=INPUT_COLUMNS):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary_path = Path(temporary_directory.name)
        input_path = temporary_path / "input.csv"
        output_path = temporary_path / "output.csv"
        with input_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        return result, output_path

    def test_prediction_runs_with_standard_library_only(self):
        result, output_path = self._run([_input_row()])
        self.assertEqual(result.returncode, 0, result.stderr)
        with output_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(list(rows[0]), INPUT_COLUMNS + ["prediction"])
        self.assertTrue(math.isfinite(float(rows[0]["prediction"])))

    def test_prediction_normalises_proportions(self):
        scaled_row = _input_row()
        scaled_row[3] = "10"
        result, output_path = self._run([_input_row(), scaled_row])
        self.assertEqual(result.returncode, 0, result.stderr)
        with output_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[0]["prediction"]), float(rows[1]["prediction"]), places=12)

    def test_invalid_input_is_rejected_without_imputation(self):
        invalid_row = _input_row(token_count="0")
        result, _ = self._run([invalid_row])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("D_tokens_B must be greater than zero", result.stderr)

        missing_columns = INPUT_COLUMNS[:-1]
        result, _ = self._run([_input_row()[:-1]], columns=missing_columns)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input CSV is missing required columns", result.stderr)


if __name__ == "__main__":
    unittest.main()
