#!/usr/bin/env python3
"""Run PPO/SAC-controlled NRBO using direct SPICE or mock feedback."""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
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
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel simulation workers per population evaluation (default: 1)",
    )
    parser.add_argument(
        "--rl-timesteps",
        type=int,
        default=64,
        help="RL environment transitions; each transition evaluates a population",
    )
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="child directory under run/<design>/")
    parser.add_argument(
        "--resume",
        help=(
            "existing run path or run ID; controller resumes from "
            "run/<design>/<run-id>/checkpoints/<PPO|SAC>/"
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--mock-failure-rate", type=float, default=0.03)
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help=(
            "keep only the N best per-sample data/ folders on disk during the run "
            "(default: keep all; e.g., --keep 50 keeps the best 50, metrics remain in results.csv/json)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase progress detail (-v progress, -vv debug)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)],
        handlers=[logging.StreamHandler()],
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    started = time.perf_counter()
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    full_command = shlex.join(["python3", "optimize_sizing.py", *cli_args])
    run_dir = run_rl_sizing(
        netlist=args.netlist,
        runner_name=args.runner,
        algorithm=args.algorithm,
        samples=args.samples,
        seed=args.seed,
        population_size=args.population_size,
        workers=args.workers,
        timesteps=args.rl_timesteps,
        episode_length=args.episode_length,
        mock_failure_rate=args.mock_failure_rate,
        run_id=args.run_id,
        resume=args.resume,
        timeout_s=args.timeout_s,
        keep=args.keep,
        command=full_command,
        verbose=args.verbose,
    )
    print(f"Results: {run_dir}")
    print(f"Elapsed: {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
