# RL-Controlled Analog Sizing

This branch runs reinforcement learning controllers over a fixed NRBO metaheuristic. PPO or
SAC selects the NRBO restart rate, Newton/population-move balance, and step radius during
search. Every candidate is evaluated directly by the selected simulator; no supervised
surrogate model or pre-collected dataset is used.

```text
ai_in_analog/
  optimize_sizing.py       PPO/SAC + NRBO entry point
  netlists/<design>/       netlist.spice and sizing.yaml
  src/rl_nrbo.py           direct-feedback RL controller and NRBO population search
  src/rl_workflow.py       simulator evaluation, objective/reward, and run artifacts
  runner/                  ngspice and mock runners
  scripts/                 SPICE deck generation and metric extraction
  run/<design>/<run-id>/   policy, candidates, and measured sample records
```

## Setup

Install ngspice and the configured SKY130A PDK using [env_setup.md](env_setup.md). Then
create the Python environment from the repository root:

```bash
bash setup_env.sh
source .venv/bin/activate
```

`requirements.txt` contains the RL and simulator-pipeline dependencies, including
Stable-Baselines3, Gymnasium, and PyTorch. The PDK and ngspice are system tools and are not
installed by pip.

## Run PPO + NRBO

```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --algorithm PPO \
  --samples 5 \
  --population-size 4 \
  --rl-timesteps 64 \
  --episode-length 8 \
  --seed 0
```

## Run SAC + NRBO

```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --algorithm SAC \
  --samples 5 \
  --population-size 4 \
  --rl-timesteps 64 \
  --episode-length 8 \
  --seed 0
```

PPO is on-policy; SAC is off-policy. Both learn operator controls from measured reward:
the weighted candidate-score improvement and the fraction of feasible candidates. Score
weights, metric references, and `constraints.max_power_w` are configured in
`netlists/<design>/scoring.yaml`; no objective or weight flags are accepted by the CLI.

### Score configuration

For a design, editable score weights and references are stored in
`netlists/<design>/scoring.yaml`. The StrongARM defaults are:

```yaml
weights:
  delay: 1.0
  speed: 0.0
  power: 0.0
  energy: 0.0
  offset: 0.0
  noise: 0.0
  kickback: 0.0
references:
  delay_s: 5.0e-10
  power_w: 1.0e-2
  energy_j: 1.0e-15
  offset_v: 1.0e-3
  noise_v: 1.0e-3
  kickback_v: 1.0e-3
constraints:
  max_power_w: null
```

The score is read only from `netlists/<design>/scoring.yaml`. The resolved
weights/references are stored in `candidates.json`, `results.json`, and
`run/<design>/<run-id>/scoring.yaml`.

### Score formula

The NRBO candidate score is minimized. There is no fixed initial or final score range; it
depends on the selected units and weights. The implementation uses:

$$
S =
w_d\frac{t_d}{T_{ref}}
-w_s(f_sT_{ref})
+w_p\frac{P}{P_{ref}}
+w_e\frac{E}{E_{ref}}
+w_o\frac{|V_{offset}|}{V_{offset,ref}}
+w_n\frac{V_{noise}}{V_{noise,ref}}
+w_k\frac{V_{kickback}}{V_{kickback,ref}}
+100\left[\max\left(0,\frac{P}{P_{max}}-1\right)\right]^2
$$

where $t_d$ is delay, $f_s=1/t_d$ is speed, $P$ is power, $E$ is energy per comparison,
$V_{offset}$ is input-referred offset magnitude, $V_{noise}$ is input-referred noise,
and $V_{kickback}$ is kickback noise. $T_{ref}$ is `analyses.transient.period_s`.
All references and weights come from `scoring.yaml`. Missing weighted metrics receive a
$1000$ score sentinel.

Single-objective CLI selection has been removed. Every candidate is scored from the
weighted terms in `scoring.yaml`; set the weights to select and balance the desired metrics.
Failed candidates receive $1000$, and
candidates missing an objective value receive $1000$ plus the power penalty, if applicable.
The penalty term can raise their score substantially, while infeasible candidates are kept in
results but sorted after feasible candidates.

The RL reward is separate from the candidate score:

$$
R=2\tanh\left(\frac{S_{previous}-S_{best}}{|S_{initial}|+10^{-6}}\right)
+0.05F-0.01
$$

where $F$ is the fraction of the population that satisfies the power constraint and
$S_{previous}-S_{best}$ is improvement in the candidate score. With the current reward
constants, $R$ is approximately in $[-0.01,2.04]$, while candidate scores have no fixed
bounded range.

Available measured targets are `delay_s`, `speed_hz`, `power_w`, and
`energy_per_comparison_j`. Offset, input-referred noise, and kickback require a dedicated
real-SPICE analysis recipe: the mock runner provides synthetic values for pipeline testing,
but ngspice reports these fields only when their measurement is configured and extracted.

Use `-v` for start, candidate-evaluation, checkpoint, ranking, and completion INFO logs.
Use `-vv` for additional DEBUG details supported by the workflow. `-v` does not change
how many samples are evaluated.

Default run IDs use `run_YYYY_MM_DD_HH_MM`, for example
`run/strongarm/run_2026_11_20_17_07/`. Passing `--run-id test` creates
`run/strongarm/run_test/`; passing `--run-id run_test` does not add a duplicate prefix.

Each RL environment transition evaluates a population with SPICE. PPO/SAC rollout collection
can therefore require many simulator calls: increasing `--population-size`,
`--rl-timesteps`, or `--episode-length` increases runtime and simulation count. Start with
small values to validate a design setup.

Use `-v` to print start, candidate evaluation, checkpoint, ranking, and completion progress.
Use `-vv` to add resolved scoring configuration and score-config path details.
Default run IDs use `run_YYYY_MM_DD_HH_MM`; for example:
`run/strongarm/run_2026_11_20_17_07/`. A custom `--run-id test` becomes `run_test`; a
custom value already starting with `run_` is not prefixed again.

The run directory contains `candidates.json`/`candidates.csv`,
`results.json`/`results.csv`, the sizing config snapshot, and per-evaluation decks, logs,
and metrics. Candidate rankings are based on direct simulator measurements, not surrogate
predictions.

## Resume a checkpoint

Each run stores controller checkpoints beneath:

```text
run/<design>/<run-id>/checkpoints/
  PPO/
    model.zip
  SAC/
    model.zip
    replay_buffer.pkl
```

Resume a run by run ID or full path:

```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --algorithm PPO \
  --resume run_resume_smoke \
  --rl-timesteps 64 \
  --episode-length 8 \
  --seed 102
```

The run ID and algorithm must match the existing run. Resume continues in the same run
folder, retains existing measurements, continues sample numbering, and replaces the
controller checkpoint after further training.

For a pipeline smoke test without ngspice, use the synthetic mock runner:

```bash
python3 optimize_sizing.py \
  --runner mock \
  --netlist netlists/strongarm/ \
  --algorithm PPO \
  --samples 3 \
  --population-size 4 \
  --rl-timesteps 64 \
  --episode-length 4 \
  --seed 0
```

Mock metrics are synthetic and are only for checking the RL/search/dataflow. This branch
supports PPO and SAC with NRBO; standalone NRBO, supervised regression, MBPO, GRPO, and
other RL controllers are not implemented.
