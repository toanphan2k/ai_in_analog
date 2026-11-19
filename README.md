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
  --delay-weight 1.0 \
  --power-weight 0.25 \
  --max-power-w 0.02 \
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
  --delay-weight 1.0 \
  --power-weight 0.25 \
  --max-power-w 0.02 \
  --seed 0
```

PPO is on-policy; SAC is off-policy. Both learn operator controls from measured reward:
normalized delay/power objective improvement and the fraction of feasible candidates. Set
`--objective delay_s`, `speed_hz`, or `power_w` to optimize a single measured target instead
of using weights. `--maximize-objective` reverses the selected objective direction. A power
constraint requires a measured `power_w` value.

Each RL environment transition evaluates a population with SPICE. PPO/SAC rollout collection
can therefore require many simulator calls: increasing `--population-size`,
`--rl-timesteps`, or `--episode-length` increases runtime and simulation count. Start with
small values to validate a design setup.

The run directory contains `rl_controller.zip`, `candidates.json`/`candidates.csv`,
`results.json`/`results.csv`, the sizing config snapshot, and per-evaluation decks, logs,
and metrics. Candidate rankings are based on direct simulator measurements, not surrogate
predictions.

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
