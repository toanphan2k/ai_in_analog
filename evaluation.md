# StrongARM RL-NRBO Evaluation Guide (SAC & PPO)

This guide provides the standard evaluation matrix and ready-to-run commands for **SAC** and **PPO** using the default **`--workers 8`** and **`--population-size 8`** across **3 seeds (`100`, `200`, `300`)**, along with post-run plotting and candidate validation.

---

## 1. Evaluation Budget Table (Default: `P=8`, `W=8`, Both Algorithms, Seeds `100, 200, 300`)

Because every `--episode-length` below is a multiple of `8` (PPO's rollout length), **SAC and PPO evaluate the exact same number of SPICE circuits** at every tier, ensuring a strictly fair comparison.

Runtime estimates are calibrated to measured throughput on `W=8` (~`300` SPICE evals/min; `21,120` evals = `71 min`).

| Tier | `--episode-length` | `--rl-timesteps` | Episodes (+1 Rollout) | Exact Evals / Run | Est. Time / Run (`W=8`) | Total Time (2 Algos × 3 Seeds = 6 Runs) | Disk / Run (`--keep 150`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard (2 GHz / Fast) | `24` | `1440` | `60 + 1` | **`10,744`** | ~36 min | ~3.6 h | ~500 MB *(~170 MB w/ `--keep 50`)* |
| Fast 5 GHz Benchmark | `32` | `2880` | `90 + 1` | **`21,120`** | ~71 min *(measured)* | ~7.1 h | ~500 MB |
| **Best Recommended (5 GHz)** | **`48`** | **`5760`** | **`120 + 1`** | **`41,632`** | **~2.3 h** | **~14.0 h** | **~515 MB** |
| Exhaustive | `64` | `12800` | `200 + 1` | **`91,664`** | ~5.1 h | ~30.7 h | ~545 MB |

### 3-Seed Run Matrix (Both Algorithms, Default `W=8`)

| Run # | Algorithm | Seed | `--workers` | `--population-size` | `--episode-length` | `--rl-timesteps` | `--keep` | Run Directory | Best Feasible Score | 45-Corner PVT Pass |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | **SAC** | `100` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |
| 2 | **SAC** | `200` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |
| 3 | **SAC** | `300` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |
| 4 | **PPO** | `100` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |
| 5 | **PPO** | `200` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |
| 6 | **PPO** | `300` | `8` | `8` | `48` | `5760` | `150` | _TBD_ | _TBD_ | _TBD_ |

> **Why `--population-size 8` with `--workers 8`?**
> Every `reset()` (`8` candidates) and every `step()` (`7` new candidates + `1` cached elite) executes in **1 parallel wave** across 8 CPU cores while giving NRBO proper top-20% elite diversity.
>
> **Why keep `scoring.yaml` fixed across all 6 runs?**
> Keep [netlists/strongarm_5ghz/scoring.yaml](netlists/strongarm_5ghz/scoring.yaml) identical across seeds `100, 200, 300` for both SAC and PPO so `mean ± std` and best-score comparisons are strictly apples-to-apples.

---

## 2. Activate Environment

```bash
source .venv/bin/activate
```

---

## 3. 5 GHz StrongARM (`netlists/strongarm_5ghz/`)

### Run All 6 Cases Automatically (SAC & PPO × Seeds `100, 200, 300`)
```bash
for algo in SAC PPO; do
  for seed in 100 200 300; do
    python3 optimize_sizing.py \
      --runner ngspice \
      --netlist netlists/strongarm_5ghz/ \
      --algorithm "$algo" \
      --population-size 8 \
      --workers 8 \
      --episode-length 48 \
      --rl-timesteps 5760 \
      --samples 20 \
      --keep 150 \
      --seed "$seed" \
      -v
  done
done
```

### Single-Seed Commands (5 GHz, replace `--seed` with `100`, `200`, or `300`)

#### SAC (5 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_5ghz/ \
  --algorithm SAC \
  --population-size 8 \
  --workers 8 \
  --episode-length 48 \
  --rl-timesteps 5760 \
  --samples 20 \
  --keep 150 \
  --seed 100 \
  -v
```

#### PPO (5 GHz)
```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm_5ghz/ \
  --algorithm PPO \
  --population-size 8 \
  --workers 8 \
  --episode-length 48 \
  --rl-timesteps 5760 \
  --samples 20 \
  --keep 150 \
  --seed 100 \
  -v
```

---

## 4. 2 GHz StrongARM (`netlists/strongarm_2ghz/`)

### Run All 6 Cases Automatically (SAC & PPO × Seeds `100, 200, 300`)
```bash
for algo in SAC PPO; do
  for seed in 100 200 300; do
    python3 optimize_sizing.py \
      --runner ngspice \
      --netlist netlists/strongarm_2ghz/ \
      --algorithm "$algo" \
      --population-size 8 \
      --workers 8 \
      --episode-length 24 \
      --rl-timesteps 1440 \
      --samples 10 \
      --keep 50 \
      --seed "$seed" \
      -v
  done
done
```

### Single-Seed Commands (2 GHz, replace `--seed` with `100`, `200`, or `300`)

#### SAC (2 GHz)
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
  --seed 100 \
  -v
```

#### PPO (2 GHz)
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
  --seed 100 \
  -v
```

---

## 5. Plotting Results (Single Run & SAC vs. PPO Comparison)

```bash
# Single run (generates Plots 1–5 + top-5 transient waveforms)
python3 plot/plot.py --output-dir run/strongarm_5ghz/<sac-run-id>
python3 plot/plot.py --output-dir run/strongarm_5ghz/<ppo-run-id>

# Side-by-side SAC vs. PPO comparison (e.g., matching seed 100, 200, or 300)
python3 plot/plot.py \
  --output-dir run/strongarm_5ghz/<sac-seed100-run-id> run/strongarm_5ghz/<ppo-seed100-run-id>
```

Plots are saved automatically to:
```text
plot_<algorithm>_seed_<seed>_iteration_<iteration>_<run-time-id>/
```

---

## 6. Post-Search Validation (Small-Input Overdrive, PVT Corners, MC Offset)

Validate the top-5 candidates from `candidates.csv` across `±1/5/10/100 mV` input overdrive, 45 PVT corners (`tt, ff, ss, sf, fs` × `0/27/85 °C` × `VDD ± 10%`), and Monte-Carlo mismatch offset bisection (default `--workers 8`):

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

You can also validate specific sample IDs from `results.csv` without overwriting `validation/`:
```bash
python3 scripts/validate_candidates.py \
  --output-dir run/strongarm_5ghz/<run-id> \
  --samples 015067 013900 \
  --val-subdir validation_custom \
  --workers 8 \
  --mc-samples 24
```

Outputs are written to `run/<design>/<run-id>/validation/`:
- `validation_summary.csv`
- `small_input_delay.csv`
- `pvt_corners.csv`
- `monte_carlo_offset.csv`
