import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class JointPredictionTests(unittest.TestCase):
    def test_b_prediction_runs_without_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "b_input.csv"
            output_path = temporary_path / "b_output.csv"
            input_path.write_text(
                "N_params_B,D_tokens_B,Q_score\n1.0,300,0.8\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "joint_model.predict",
                    "--model",
                    str(ROOT / "joint_model/results/joint_model.json"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "b",
                ],
                cwd=ROOT,
                check=True,
            )
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0]["prediction"]), 0.0)

    def test_a_prediction_requires_complete_simplex(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "a_input.csv"
            output_path = temporary_path / "a_output.csv"
            columns = ["N_params_B"] + [f"p_{index}" for index in range(1, 18)]
            input_path.write_text(
                ",".join(columns) + "\n" + ",".join(["1.0", "1.0"] + ["0.0"] * 16) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "joint_model.predict",
                    "--model",
                    str(ROOT / "joint_model/results/joint_model.json"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "a",
                ],
                cwd=ROOT,
                check=True,
            )
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0]["prediction"]), 0.0)

    def test_joint_parameters_are_readable(self):
        payload = json.loads(
            (ROOT / "joint_model/results/joint_model.json").read_text(encoding="utf-8")
        )
        self.assertIn("b_only_model", payload)
        self.assertIn("p_adapter", payload)
        self.assertIn("a_adapter", payload)
        self.assertEqual(payload["a_adapter"]["p_gamma_model"]["high_n_gamma"], 0.25)


if __name__ == "__main__":
    unittest.main()
