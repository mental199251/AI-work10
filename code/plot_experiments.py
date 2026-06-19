"""Aggregate ablation outputs into report-ready tables and plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from ddpm_common import ensure_dir, load_json, write_csv
from project_cli import project_path


def read_last_metric(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default="outputs/experiments")
    parser.add_argument("--output-dir", default="outputs/experiment_summary")
    args = parser.parse_args()
    experiment_root = project_path(args.experiment_root)
    output_dir = ensure_dir(project_path(args.output_dir))
    summaries: List[Dict[str, Any]] = []
    for summary_path in sorted(experiment_root.glob("*/*/training_summary.json")):
        run_dir = summary_path.parent
        config = load_json(run_dir / "config.json")
        summary = load_json(summary_path)
        last = read_last_metric(run_dir / "metrics.csv")
        summaries.append(
            {
                "dimension": run_dir.parent.name,
                "experiment": run_dir.name,
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "timesteps": config["timesteps"],
                "base_channels": config["base_channels"],
                "max_train_samples": config["max_train_samples"],
                "final_loss": float(last.get("loss", "nan")),
                "best_loss": float(summary["best_loss"]),
                "training_seconds": float(summary["total_seconds_this_run"]),
                "parameter_count": int(summary["parameter_count"]),
            }
        )
    if not summaries:
        raise FileNotFoundError(f"No completed experiments found under {experiment_root}")
    write_csv(summaries, output_dir / "experiment_comparison.csv")
    for dimension in sorted({row["dimension"] for row in summaries}):
        rows = [row for row in summaries if row["dimension"] == dimension]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        names = [row["experiment"] for row in rows]
        axes[0].bar(names, [row["best_loss"] for row in rows], color="#59A14F")
        axes[0].set_ylabel("Best MSE loss")
        axes[1].bar(names, [row["training_seconds"] for row in rows], color="#F28E2B")
        axes[1].set_ylabel("Training time (seconds)")
        for axis in axes:
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"Controlled experiment: {dimension}")
        figure.tight_layout()
        figure.savefig(output_dir / f"comparison_{dimension}.png", dpi=180)
        plt.close(figure)
    print(f"Experiment summary: {output_dir}")


if __name__ == "__main__":
    main()

