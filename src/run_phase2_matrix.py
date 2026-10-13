"""Launch the main, budget, or GRO Phase 2 experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MAIN_SEEDS = (42, 101, 202)
BUDGETS = (100, 250, 500, 1000, 2000, 3000)
RHO_VALUES = (0.01, 0.05, 0.10, 0.20)
ALPHA_VALUES = (0.20, 0.50, 0.80)


def parse_numbers(value: str, cast: type) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def jobs_for_mode(mode: str) -> list[tuple[str, int, float, float]]:
    """Return ``(sweep, budget, rho, alpha)`` experiment settings."""
    if mode == "main":
        return [("main", 2000, 0.05, 0.20)]
    if mode == "budget":
        return [("budget", budget, 0.05, 0.20) for budget in BUDGETS]
    jobs = [
        ("rho", budget, rho, 0.50)
        for budget in (1000, 2000)
        for rho in RHO_VALUES
    ]
    jobs.extend(
        ("alpha", budget, 0.05, alpha)
        for budget in (1000, 2000)
        for alpha in ALPHA_VALUES
    )
    unique: dict[tuple[int, float, float], tuple[str, int, float, float]] = {}
    for job in jobs:
        key = (job[1], job[2], job[3])
        unique.setdefault(key, job)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_paths.json"))
    parser.add_argument("--mode", choices=("main", "budget", "gro"), default="main")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase2_matrix"))
    args, passthrough = parser.parse_known_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config_base = args.config.parent
    base_model = str(payload.get("base_model", "Qwen/Qwen3-8B"))
    phase1_checkpoint = resolve(config_base, str(payload["phase1_checkpoint"]))
    dataset_config = payload.get("datasets", {})
    selected = parse_numbers(args.datasets, str) if args.datasets else list(dataset_config)
    seeds = (
        parse_numbers(args.seeds, int)
        if args.seeds
        else list(MAIN_SEEDS if args.mode == "main" else (42,))
    )

    for dataset in selected:
        if dataset not in dataset_config:
            raise KeyError(f"Dataset {dataset} is not defined in {args.config}")
        train_file = resolve(config_base, str(dataset_config[dataset]["train"]))
        dev_file = resolve(config_base, str(dataset_config[dataset]["dev"]))
        for sweep, budget, rho, alpha in jobs_for_mode(args.mode):
            for seed in seeds:
                run_name = (
                    f"{dataset}_{sweep}_n{budget}_rho{rho:g}_alpha{alpha:g}_seed{seed}"
                )
                output_dir = args.output_root / run_name
                command = [
                    sys.executable,
                    "-m",
                    "src.train_phase2",
                    "--base-model",
                    base_model,
                    "--phase1-checkpoint",
                    str(phase1_checkpoint),
                    "--train-file",
                    str(train_file),
                    "--dev-file",
                    str(dev_file),
                    "--output-dir",
                    str(output_dir),
                    "--maximum-train-samples",
                    str(budget),
                    "--rho",
                    str(rho),
                    "--alpha",
                    str(alpha),
                    "--seed",
                    str(seed),
                    *passthrough,
                ]
                print(f"Running {run_name}")
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
