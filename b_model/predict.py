from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

try:
    from b_model.pipeline import RESULTS_DIR, predict_frame
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from b_model.pipeline import RESULTS_DIR, predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict validation Loss with the frozen B-only model")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing N_params_B, D_tokens_B and optional Q_score")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument("--model", type=Path, default=RESULTS_DIR / "B_only_model.json")
    args = parser.parse_args()

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    prediction = predict_frame(payload, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
