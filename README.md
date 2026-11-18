# AI-assisted sizing

The project separates design-space/AI logic from simulator and netlist helpers:

```text
ai_in_analog/
  netlists/<design>/   netlist.spice and sizing.yaml
  src/                 candidate selection and dataset/AI logic only
  scripts/             parameter parsing, deck generation, and metric extraction
  runner/              simulator adapters
  optimize_sizing.py   generic sizing and evaluation entry point
  run/<run-id>/        generated decks, simulator output, CSV, and JSON
```

Each design folder supplies its sizing variable names and ranges, transistor groups,
SPICE `.param` mapping, and enabled analyses in `sizing.yaml`. Deck generation and
measurement scripts consume this config rather than embedding StrongARM instance names.

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

The netlist directory name is combined with a numeric UTC timestamp to form the default
output path, for example `run/strongarm_20261118150023990787/`. Choose `--run-id` to
set the output folder name explicitly; existing non-empty output directories are never
overwritten. `--algorithm` supports `halton` (space-filling initial design) and `random`;
`--samples`, `--seed`, and `--timeout-s` control the run.

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
