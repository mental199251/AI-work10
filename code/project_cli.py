"""Consistent command-line handling for training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ddpm_common import load_json


PROJECT_DIR = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def training_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--beta-schedule", choices=("linear", "cosine"))
    parser.add_argument("--beta-start", type=float)
    parser.add_argument("--beta-end", type=float)
    parser.add_argument("--cosine-s", type=float)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--label-dropout", type=float)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--preview-sampling-steps", type=int)
    parser.add_argument("--preview-guidance-scale", type=float)
    parser.add_argument("--preview-images", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--resume", help="Checkpoint path, or 'auto' for output/latest.pt")
    parser.add_argument("--cifar-archive")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument("--download", dest="download", action="store_true")
    download_group.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=None)
    return parser


def resolve_training_config(
    parser: argparse.ArgumentParser,
    dataset: str,
    conditional: bool,
) -> Dict[str, Any]:
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_json(config_path)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key.replace("-", "_")] = value
    config["dataset"] = dataset
    config["conditional"] = conditional
    config["config_source"] = str(config_path)
    for key in ("data_dir", "output_dir", "cifar_archive"):
        if config.get(key):
            config[key] = str(project_path(str(config[key])))
    if config.get("resume") and config["resume"] != "auto":
        config["resume"] = str(project_path(str(config["resume"])))
    return config
