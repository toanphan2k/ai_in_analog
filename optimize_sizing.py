#!/usr/bin/env python3
"""Run PPO/SAC-controlled NRBO using direct SPICE or mock feedback."""
from __future__ import annotations

import argparse
import logging
import time
from typing import Sequence

from src.rl_workflow import run_rl_sizing


def main(argv: Sequence[str] | None = None) -> int:
    """Parse RL-NRBO options, execute the search, and print its output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("ngspice", "mock"), default="ngspice")
    parser.add_argument(
        "--netlist",
        required=True,
        help="design directory containing netlist.spice and sizing.yaml",
    )
    parser.add_argument(
        "--algorithm",
        type=str.upper,
        choices=("PPO", "SAC"),
        default="PPO",
        help="RL controller; NRBO is always the metaheuristic",
    )
    parser.add_argument("--samples", type=int, default=5, help="number of top measured candidates")
    parser.add_argument(
        "--objective",
        choices=("delay_s", "speed_hz", "power_w"),
        help="single measured objective; otherwise use weighted objectives",
    )
    parser.add_argument("--maximize-objective", action="store_true")
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=0.0)
    parser.add_argument("--power-weight", type=float, default=0.0)
    parser.add_argument("--power-reference-w", type=float, default=0.01)
    parser.add_argument("--max-power-w", type=float)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument(
        "--rl-timesteps",
        type=int,
        default=64,
        help="RL environment transitions; each transition evaluates a population",
    )
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="child directory under run/<design>/")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--mock-failure-rate", type=float, default=0.03)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)],
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    started = time.perf_counter()
    run_dir = run_rl_sizing(
        netlist=args.netlist,
        runner_name=args.runner,
        algorithm=args.algorithm,
        samples=args.samples,
        seed=args.seed,
        objective=args.objective,
        maximize_objective=args.maximize_objective,
        delay_weight=args.delay_weight,
        speed_weight=args.speed_weight,
        power_weight=args.power_weight,
        power_reference_w=args.power_reference_w,
        max_power_w=args.max_power_w,
        population_size=args.population_size,
        timesteps=args.rl_timesteps,
        episode_length=args.episode_length,
        mock_failure_rate=args.mock_failure_rate,
        run_id=args.run_id,
        timeout_s=args.timeout_s,
    )
    print(f"Results: {run_dir}")
    print(f"Elapsed: {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
