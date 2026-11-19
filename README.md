# AI-assisted sizing

The project separates design-space/AI logic from simulator and netlist helpers:

```text
ai_in_analog/
  netlists/<design>/   netlist.spice and sizing.yaml
  src/                 design features, candidate selection, and NRBO search logic
  scripts/             parameter parsing, deck generation, and metric extraction
  runner/              simulator adapters
  collect_data.py     space-filling SPICE dataset collection
  train_surrogate.py  surrogate model training and evaluation
  optimize_sizing.py  generic sizing and evaluation entry point
  run/<design>/<run-id>/ generated decks, simulator output, CSV, and JSON
  dataset/<design>/     collected CSV or Parquet datasets
```

Each design folder supplies sizing variables, physical bounds and scale, transistor groups,
SPICE `.param` mapping, derived geometry features, and enabled analyses in `sizing.yaml`.
Symmetric devices are represented by one group-level W/L pair. Dataset feature engineering
adds each group's W/L and W*L while retaining the physical sizing variables.

## Setup and run

```bash
bash setup_env.sh
source .venv/bin/activate
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --samples 10 \
  --algorithm halton \
  --seed 0
```

## Code Quality

Install the development linters into the project environment and run them across all
Python entry points and packages:

```bash
python3 -m pip install -r requirements-dev.txt
flake8 collect_data.py optimize_sizing.py train_surrogate.py src runner scripts
pylint collect_data.py optimize_sizing.py train_surrogate.py src runner scripts --score=n
```

Flake8 and Pylint use the repository `.flake8` and `.pylintrc` settings. In VS Code,
select `.venv/bin/python` on Linux/WSL or `.venv\Scripts\python.exe` on Windows for
Pylance so optional and runtime imports resolve against the project environment.

## Collect and train

Collect a space-filling dataset by running the same configured SPICE analyses used by
`optimize_sizing.py`. `lhs` is the default; `sobol`, `halton`, and `random` are also
available. Collection writes both the normal run folder and a flattened dataset:

```bash
python3 collect_data.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --samples 100 \
  --algorithm lhs \
  --seed 0
```

By default, the dataset is written to `dataset/<netlist-name>/data.csv` (for example,
`dataset/strongarm/data.csv`). The generic `--output dataset/data.csv` path is treated as
the same design-scoped default; other explicit output paths are used as provided.

Use a `.parquet` output path for Parquet export; this requires optional `pandas` and
`pyarrow` packages. Failed and partial simulations are retained with their status and
available measurements. Sizing columns are named `size__<variable>` and measured targets
retain the metric names from the run results.

For a pipeline smoke test on a machine without ngspice, choose the deterministic mock
runner. Its measurements are synthetic and are not suitable for circuit design decisions:

```bash
python3 collect_data.py \
  --runner mock \
  --netlist netlists/strongarm/ \
  --samples 40 \
  --algorithm lhs \
  --seed 23 \
  --mock-failure-rate 0.05 \
  --output dataset/strongarm/mock_data.csv
```

Train on the collected dataset and save a model plus a held-out evaluation report:

```bash
python3 train_surrogate.py \
  --dataset dataset/strongarm/data.csv \
  --method gpr \
  --targets delay_s,speed_hz,power_w \
  --output model.pkl \
  --target-transform log-standard
```

Training supports `gpr` (Gaussian Process with Matérn 5/2 ARD), `lightgbm`, `catboost`,
`mlp`, and `two-stage` (validity classifier followed by regression). LightGBM and CatBoost
are included in `requirements.txt`. Targets default to standardized
log-space; use `--target-transform standard` or `none` when appropriate. Failed samples
are excluded from regression by default and remain labeled for two-stage classification;
`--include-partial` opts partial results into regression. The model is serialized with
joblib. `<model>_evaluation.json` reports held-out MAE, R2, MAPE, and feature importance
(GPR ARD relevance and permutation importance). Only use model files from trusted sources
when loading serialized artifacts.

## Surrogate optimization

`optimize_sizing.py --algorithm nrbo` loads a trained artifact and searches the configured
sizing bounds using a finite-difference Newton-Raphson population search with randomized
escape moves. It minimizes normalized weighted delay, speed, and power objectives; defaults
minimize delay only. `speed_hz` is derived as `1 / delay_s` during collection, and a positive
speed weight maximizes predicted speed. An explicit `--objective` selects one model target;
use `--maximize-objective` for targets where larger is better. An optional predicted power
limit adds a constraint penalty. With a two-stage model, `--min-valid-probability` sets the
required predicted validity. In one invocation, NRBO loads the model, proposes candidates,
and evaluates them using the selected runner. It saves surrogate proposals and measured
results under `run/<design>/<run-id>/`.

```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --samples 5 \
  --algorithm nrbo \
  --model model.pkl \
  --delay-weight 0.0 \
  --speed-weight 1.0 \
  --power-weight 0.25 \
  --max-power-w 0.02 \
  --min-valid-probability 0.8 \
  --seed 0
```

`--samples` sets the number of candidate sizings returned in NRBO mode. Weighted mode
requires the targets corresponding to nonzero weights; `--max-power-w` also requires a
trained `power_w` target. The search implementation is local to this project
and uses numerical gradients/curvature plus population diversity; it is not a drop-in
implementation of every published NRBO variant. `--neighbors-per-candidate` optionally
adds local sizings to the same evaluation batch. qNEHVI/MOBO is not currently implemented.

The NRBO run writes `candidates.csv`/`candidates.json` and the usual `results.csv`/`results.json`.
Append the measured results to the training dataset and retrain:

```bash
python3 collect_data.py \
  --netlist netlists/strongarm/ \
  --from-results run/strongarm/<nrbo-run-id>/results.json \
  --output dataset/data.csv \
  --append

python3 train_surrogate.py \
  --dataset dataset/strongarm/data.csv \
  --method gpr \
  --targets delay_s,power_w \
  --output model.pkl
```

Use `--algorithm verify --candidates <path>` only when you want to run a previously saved
candidate file in a separate invocation.

The netlist directory name forms the circuit folder under `run/`; each run uses a numeric
UTC timestamp as the default run ID, for example
`run/strongarm/20261118150023990787/`. Choose `--run-id` to set the child folder explicitly;
existing non-empty output directories are never overwritten. `--algorithm` supports `halton`,
`random`, `lhs`, `sobol`, `nrbo`, and `verify`;
`nrbo` requires `--model`, while `verify` requires `--candidates`. `--samples`, `--seed`,
and `--timeout-s` control the run.

Use `-v` for progress and metrics, `-vv` for sizing and simulator-command details, and
`-vvv` for generated deck contents and ngspice logs.

For each sampled W/L design, a sample folder contains a sizing JSON, a generated transient
deck and waveform, ngspice log, and—when enabled by the design config—an AC gain deck,
waveform, and log. It also contains a `result.json`. The run folder contains `results.csv`,
the copied sizing config, and `results.json` with the run name and all sample records.
The `netlists/<design>/sizing.yaml` file is the editable source config; the run-folder copy
is a snapshot of the exact settings used for that run, so results remain reproducible if
the source config changes later.

Each sample JSON includes a `name`, a `size` dictionary, normalized vector, extracted
metrics, status, and paths to generated files. Failed analyses remain recorded with an
error instead of being treated as valid results. Offset and kickback fields are only
present when the selected netlist config supplies those analysis recipes.

For a different circuit, add `netlists/<design>/netlist.spice` and a matching
`netlists/<design>/sizing.yaml`; no StrongARM paths or transistor labels are required by the
optimizer itself.
