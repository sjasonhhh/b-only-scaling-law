# Internal A+B model

This directory contains the internal joint candidate. It does not replace the public B-only interface or `INTERFACE.md`.

## Fit and evaluate

```bash
PYTHONPATH=/path/to/b-scaling-law/.venv/lib/python3.12/site-packages:/path/to/b-scaling-law \
python3 tools/fit_joint_model.py \
  --derived-dir data/derived \
  --b-only-model b_model/results/B_only_model.json \
  --output-dir joint_model/results
```

The fitting command requires the optional scientific Python dependencies listed in `requirements.txt`. The frozen prediction entrypoint in `joint_model/predict.py` itself uses only the standard library.

The A+B model is intentionally internal until the A-side input/output contract is confirmed. Hidden labels and the private evaluator are outside this directory.
