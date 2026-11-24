# StrongARM RL-NRBO Evaluation Guide (SAC & PPO)

This guide provides ready-to-run commands for **SAC** and **PPO** with `--episode-length 24`, `--population-size 8`, and `--workers 8`, along with post-run plotting and candidate validation.

---

## 1. Recommended `--rl-timesteps` Budgets (`ep=24`, `P=8`, `W=8`)

Because `24` is a multiple of `8` (PPO's rollout length), **SAC and PPO evaluate the exact same number of SPICE circuits** at every multiple of `24`, ensuring a strictly fair comparison:

| Tier | `--rl-timesteps` | Training Episodes | Final Rollout | Exact SPICE Evals | Est. Runtime (`W=8`) | Disk (`--keep 50`) |
|---|---:|---:|---:|---:|---:|---:|
| Standard | `960` | `40` | `+1` (`24` steps) | **`7,224`** | ~23–35 min | ~170 MB |
| **Best Recommended** | **`1440`** | **`60`** | **`+1` (`24` steps)** | **`10,744`** | **~35–50 min** | **~170 MB** |
| Deep / Exhaustive | `2880` | `120` | `+1` (`24` steps) | **`21,304`** | ~70–100 min | ~170 MB |

> **Why `--population-size 8` with `--workers 8`?**
> Every `reset()` (`8` candidates) and every `step()` (`7` new candidates + `1` cached elite) executes in **1 parallel wave** across 8 CPU cores while giving NRBO proper top-20% elite diversity.
>
> **Why `--rl-timesteps 1440`?**
> Gives `60` full 24-step episodes (`10,744` evaluations) — enough episodes for both off-policy **SAC** and on-policy **PPO** (`180` policy updates) to converge cleanly, while finishing in ~35–50 minutes per run thanks to the reduced SKY130 corner models and transient early-exit.

---

## 2. Activate Environment

```bash
source .venv/bin/activate
```

---

## 3. 5 GHz StrongARM (`netlists/strongarm_5ghz/`)

### Run SAC (5 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_5ghz/ \
  --algorithm SAC \
  --population-size 8 \
  --workers 8 \
  --episode-length 24 \
  --rl-timesteps 1440 \
  --samples 10 \
  --keep 50 \
  --seed 0 \
  -v
```

### Run PPO (5 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_5ghz/ \
  --algorithm PPO \
  --population-size 8 \
  --workers 8 \
  --episode-length 24 \
  --rl-timesteps 1440 \
  --samples 10 \
  --keep 50 \
  --seed 0 \
  -v
```

---

## 4. 2 GHz StrongARM (`netlists/strongarm_2ghz/`)

### Run SAC (2 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_2ghz/ \
  --algorithm SAC \
  --population-size 8 \
  --workers 8 \
  --episode-length 24 \
  --rl-timesteps 1440 \
  --samples 10 \
  --keep 50 \
  --seed 0 \
  -v
```

### Run PPO (2 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_2ghz/ \
  --algorithm PPO \
  --population-size 8 \
  --workers 8 \
  --episode-length 24 \
  --rl-timesteps 1440 \
  --samples 10 \
  --keep 50 \
  --seed 0 \
  -v
```

---

## 5. Plotting Results (Single Run & SAC vs. PPO Comparison)

```bash
# Single run (generates Plots 1–5 + top-5 transient waveforms)
python3 plot/plot.py --output-dir run/strongarm_5ghz/<sac-run-id>
python3 plot/plot.py --output-dir run/strongarm_5ghz/<ppo-run-id>

# Side-by-side SAC vs. PPO comparison on the same axes
python3 plot/plot.py \
  --output-dir run/strongarm_5ghz/<sac-run-id> run/strongarm_5ghz/<ppo-run-id>
```

Plots are saved automatically to:
```text
plot_<algorithm>_seed_<seed>_iteration_<iteration>_<run-time-id>/
```

---

## 6. Post-Search Validation (Small-Input Overdrive, PVT Corners, MC Offset)

Validate the top-5 candidates from `candidates.csv` across `±1/5/10/100 mV` input overdrive, 45 PVT corners (`tt, ff, ss, sf, fs` × `0/27/85 °C` × `VDD ± 10%`), and Monte-Carlo mismatch offset bisection:

```bash
python3 scripts/validate_candidates.py \
  --output-dir run/strongarm_5ghz/<sac-run-id> \
  --top-k 5 \
  --workers 8 \
  --mc-samples 24

python3 scripts/validate_candidates.py \
  --output-dir run/strongarm_5ghz/<ppo-run-id> \
  --top-k 5 \
  --workers 8 \
  --mc-samples 24
```

Outputs are written to `run/<design>/<run-id>/validation/`:
- `validation_summary.csv`
- `small_input_delay.csv`
- `pvt_corners.csv`
- `monte_carlo_offset.csv`
