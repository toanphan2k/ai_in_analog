# Dataset

`collect_data.py` writes the training set to `data.csv` by default, or to Parquet when
the output filename ends in `.parquet`. Each row includes raw sizing variables, derived
per-group W/L and W*L features, metric targets, run identity, status, and quality labels.
Failed and partial rows are retained for analysis; regression excludes them by default.
A two-stage model can use their status labels for validity classification.

After candidate verification, append the SPICE results with `collect_data.py --from-results
run/<verify-run-id>/results.json --append` and retrain. The JSONL helpers in
`src/dataset.py` remain available for other append-only workflows but are not the default
format used by the collection CLI.
