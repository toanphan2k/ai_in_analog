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
  plot/                    standalone and batch plotting scripts for results.csv
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
  --workers 1 \
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
  --workers 1 \
  --rl-timesteps 64 \
  --episode-length 8 \
  --seed 0
```

PPO is on-policy; SAC is off-policy. Both learn operator controls from measured reward:
the weighted candidate-score improvement and the fraction of feasible candidates. Score
objective/sentinel, weights, metric references, constraints, and penalty parameters are configured
in `netlists/<design>/scoring.yaml`; no objective or weight flags are accepted by the CLI.

### Score configuration

For a design, editable score settings, weights, references, constraints, and penalty parameters
are stored in `netlists/<design>/scoring.yaml`. The StrongARM baseline is:

```yaml
scoring:
  objective: minimize
  infeasible_score: 1000.0

weights:
  delay: 1.0
  power: 0.5
  energy: 0.5
  offset: 1.0
  noise: 0.5
  kickback: 1.0

references:
  delay_s: 2.0e-10
  power_w: 5.0e-4
  energy_j: 1.0e-13
  offset_v: 5.0e-3
  noise_v: 5.0e-4
  kickback_v: 2.0e-2

constraints:
  max_delay_s: 3.0e-10
  max_power_w: 5.0e-4
  max_offset_v: 1.0e-2
  max_noise_v: 1.0e-3
  max_kickback_v: 2.0e-2
  min_output_swing_v: 1.5

penalty:
  lambda: 100.0
  type: quadratic
```

The score is read only from `netlists/<design>/scoring.yaml`. The resolved
scoring configuration is stored in `candidates.json`, `results.json`, and
`run/<design>/<run-id>/scoring.yaml`.

### Score formula

The NRBO candidate score $S_{\text{total}}$ is minimized ($\min S_{\text{total}}$). There is no
fixed initial or final score range; it depends on the selected units, weights, and penalties. The
implementation uses:

$$
\begin{aligned}
S_{\text{total}} = {}&
w_d \frac{t_d}{t_{d,\text{ref}}}
+ w_p \frac{P}{P_{\text{ref}}}
+ w_e \frac{E}{E_{\text{ref}}} \\
&+ w_o \frac{|V_{\text{offset}}|}{V_{\text{offset,ref}}}
+ w_n \frac{V_{\text{noise}}}{V_{\text{noise,ref}}}
+ w_k \frac{V_{\text{kickback}}}{V_{\text{kickback,ref}}} \\
&+ \lambda \left( g_P^2 + g_d^2 + g_O^2 + g_N^2 + g_K^2 + g_S^2 \right)
\end{aligned}
$$

where $t_d$ is delay, $P$ is power,
$E$ is energy per comparison, $V_{\text{offset}}$ is input-referred offset, $V_{\text{noise}}$
is input-referred noise, $V_{\text{kickback}}$ is kickback noise, and $V_{\text{swing}}$ is
output swing. All references (including $t_{d,\text{ref}}$ from `references.delay_s`), weights,
constraints, and $\lambda$ come from `scoring.yaml`. The normalized constraint violations are:

$$
g_P = \max\left(0, \frac{P}{P_{\max}} - 1\right), \quad
g_d = \max\left(0, \frac{t_d}{t_{d,\max}} - 1\right), \quad
g_O = \max\left(0, \frac{|V_{\text{offset}}|}{V_{\text{offset,max}}} - 1\right)
$$

$$
g_N = \max\left(0, \frac{V_{\text{noise}}}{V_{\text{noise,max}}} - 1\right), \quad
g_K = \max\left(0, \frac{V_{\text{kickback}}}{V_{\text{kickback,max}}} - 1\right), \quad
g_S = \max\left(0, \frac{V_{\text{swing,min}} - V_{\text{swing}}}{V_{\text{swing,min}}}\right)
$$

Single-objective CLI selection has been removed. Every candidate is scored from the
weighted terms and constraints in `scoring.yaml`; set the weights and limits to balance the
desired metrics. Failed candidates or candidates missing a weighted or constrained metric receive
`scoring.infeasible_score` (default $1000.0$) and are marked infeasible. Candidates violating any
configured constraint are marked infeasible, incur the quadratic (or linear) penalty scaled by
$\lambda$, and are sorted after feasible candidates in `candidates.json`.

The RL reward is separate from the candidate score:

$$
R=2\tanh\left(\frac{S_{previous}-S_{best}}{|S_{initial}|+10^{-6}}\right)
+0.05F-0.01
$$

where $F$ is the fraction of the population that satisfies all configured constraints and
$S_{previous}-S_{best}$ is improvement in the candidate score. With the current reward
constants, $R$ is approximately in $[-0.01,2.04]$, while candidate scores have no fixed
bounded range.

Measured targets collected in a single unified `netlist.spice` simulation per candidate
include `delay_s`, `speed_hz`, `power_w`, `energy_per_comparison_j`,
`dc_gain_v_per_v`, `bandwidth_3db_hz`, `input_referred_offset_v`,
`input_referred_noise_v`, and `kickback_noise_v`. The mock runner produces synthetic
values with the same metric schema for fast pipeline smoke tests.

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

Use `--workers N` (default `1`) to evaluate uncached candidates within each NRBO generation in
parallel while preserving deterministic sample numbering, ordering, and scoring. Setting
`--workers` up to `--population-size` (e.g., `--workers 4` for `--population-size 4`, subject to
available CPU cores) parallelizes all simulations in each reset and step generation; values above
`--population-size` provide no additional speedup.

Use `-v` to print start, candidate evaluation, checkpoint, ranking, and completion progress.
Use `-vv` to add resolved scoring configuration and score-config path details.
Default run IDs use `run_YYYY_MM_DD_HH_MM`; for example:
`run/strongarm/run_2026_11_20_17_07/`. A custom `--run-id test` becomes `run_test`; a
custom value already starting with `run_` is not prefixed again.

The run directory contains `candidates.json`, `results.json`/`results.csv`, the sizing and
scoring config snapshots, `checkpoints/`, and per-evaluation sample folders (`sample_XXXXXX/`
containing decks, logs, waveforms, and metrics) under `data/`. `results.csv` records
`sample`, `algorithm`, `iteration`, `seed`, `status`, individual design W/L columns parsed from
`sizing.yaml` (e.g., `w_in`, `l_in`, `w_n`, `w_p`, `w_rst`, `w_tail`, `w_8`, `l_8`), extracted
metrics, `rl_objective_score`, `rl_feasible`, and `error`. Candidate rankings are based on
direct simulator measurements, not surrogate predictions.

## Plotting Results

Standalone plotting scripts live under `plot/` (one script per plot, plus `plot/plot.py` to
generate all 5 plots together):

- `plot/plot_objective_score.py` — Plot 1: Objective Score vs. Evaluation Number
- `plot/plot_feasibility_rate.py` — Plot 2: Feasibility Rate vs. Evaluation Number
- `plot/plot_power_vs_delay.py` — Plot 3: Power vs. Delay (with feasible Pareto front)
- `plot/plot_offset_vs_noise.py` — Plot 4: Offset vs. Noise (with feasible Pareto front)
- `plot/plot_kickback_vs_noise.py` — Plot 5: Kickback vs. Input-Referred Noise (with feasible Pareto front)

Generate a single plot or all 5 plots from one or more `results.csv` files:

```bash
python3 plot/plot_objective_score.py --csv run/strongarm/<run-id>/results.csv
python3 plot/plot.py --csv run/strongarm/<run-id>/results.csv
```

Compare multiple runs (e.g., PPO vs. SAC or multiple seeds) on the same axes by passing multiple
CSV paths:

```bash
python3 plot/plot.py \
  --csv run/strongarm/<ppo-run>/results.csv run/strongarm/<sac-run>/results.csv
```

By default, PNGs are saved outside `run/` into the workspace directory:

```text
plot_<algorithm>_seed_<seed>_iteration_<iteration>/
```

Use `-o/--output` (for individual scripts) or `-o/--output-dir` (for `plot/plot.py`) to
override the output path, `--dpi` to adjust resolution, and `--show` to open an interactive
window.

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
