"""Run controlled MNIST parameter experiments as isolated subprocesses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ddpm_common import ensure_dir, load_json, save_json
from project_cli import PROJECT_DIR, project_path


EXPERIMENTS: Dict[str, List[Tuple[str, Any]]] = {
    "epochs": [("epochs_5", 5), ("epochs_10", 10), ("epochs_20", 20)],
    "timesteps": [("timesteps_100", 100), ("timesteps_300", 300), ("timesteps_500", 500)],
    "max_train_samples": [("samples_2000", 2000), ("samples_5000", 5000), ("samples_10000", 10000)],
    "base_channels": [("channels_16", 16), ("channels_32", 32), ("channels_48", 48)],
    "batch_size": [("batch_64", 64), ("batch_128", 128)],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/mnist_baseline.json")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=["epochs", "timesteps", "max_train_samples"],
    )
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use 2 epochs and at most 512 samples to test orchestration.")
    args = parser.parse_args()
    base = load_json(project_path(args.base_config))
    output_root = ensure_dir(project_path(args.output_root))
    config_root = ensure_dir(output_root / "configs")
    commands = []
    for dimension in args.dimensions:
        for name, value in EXPERIMENTS[dimension]:
            config = dict(base)
            config[dimension] = value
            config["output_dir"] = str(output_root / dimension / name)
            if args.quick:
                config["epochs"] = min(int(config["epochs"]), 2)
                config["max_train_samples"] = min(int(config["max_train_samples"]), 512)
                config["timesteps"] = min(int(config["timesteps"]), 50)
                config["sample_every"] = 0
            config_path = config_root / f"{dimension}_{name}.json"
            save_json(config, config_path)
            command = [
                sys.executable,
                str(PROJECT_DIR / "code" / "train_mnist_ddpm.py"),
                "--config",
                str(config_path),
            ]
            commands.append(command)
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_DIR, check=True)
    save_json({"commands": commands}, output_root / "ablation_manifest.json")


if __name__ == "__main__":
    main()

