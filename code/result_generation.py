"""Shared result generation used by MNIST and CIFAR-10 display scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from ddpm_common import (
    ensure_dir,
    save_labeled_sequence,
    save_sampling_gif,
    save_tensor_grid,
    set_seed,
)
from ddpm_data import get_dataset
from ddpm_trainer import load_model_bundle
from project_cli import project_path


def display_parser(description: str, checkpoint: str, output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", default=checkpoint)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--cifar-archive")
    return parser


def generate_standard_results(dataset_name: str, args: argparse.Namespace) -> None:
    set_seed(args.seed)
    checkpoint_path = project_path(args.checkpoint)
    output_dir = ensure_dir(project_path(args.output_dir))
    model, diffusion, checkpoint, device = load_model_bundle(checkpoint_path, args.device)
    channels = int(checkpoint["model_config"]["image_channels"])
    prefix = "mnist" if dataset_name == "mnist" else "cifar10"

    if not args.skip_forward:
        dataset, _, _ = get_dataset(
            dataset_name,
            project_path(args.data_dir),
            train=False,
            max_samples=None,
            seed=args.seed,
            download=not args.no_download,
            cifar_archive=project_path(args.cifar_archive) if args.cifar_archive else None,
        )
        image, label = dataset[0]
        image = image.unsqueeze(0).to(device)
        fixed_noise = torch.randn_like(image)
        steps = sorted(
            set(
                [
                    0,
                    diffusion.timesteps // 5,
                    diffusion.timesteps * 2 // 5,
                    diffusion.timesteps * 3 // 5,
                    diffusion.timesteps * 4 // 5,
                    diffusion.timesteps - 1,
                ]
            )
        )
        noised = []
        for step in steps:
            t = torch.tensor([step], device=device, dtype=torch.long)
            noised.append(diffusion.q_sample(image, t, fixed_noise).cpu())
        save_labeled_sequence(
            noised,
            [f"t={step}" for step in steps],
            output_dir / f"{prefix}_forward_noise_label_{label}.png",
        )

    initial_noise = torch.randn(args.num_images, channels, 32, 32, device=device)
    sampling_steps: Optional[int] = args.sampling_steps if args.sampler == "ddim" else None
    samples, frames = diffusion.sample(
        model,
        shape=initial_noise.shape,
        sampler=args.sampler,
        sampling_steps=sampling_steps,
        initial_noise=initial_noise,
        return_frames=True,
        frame_count=20,
    )
    nrow = max(1, int(args.num_images ** 0.5))
    save_tensor_grid(samples, output_dir / f"{prefix}_generated_grid.png", nrow=nrow)
    save_sampling_gif(frames, output_dir / f"{prefix}_denoising_process.gif", nrow=nrow)
    progress_labels = [f"stage {index + 1}/{len(frames)}" for index in range(len(frames))]
    save_labeled_sequence(
        frames,
        progress_labels,
        output_dir / f"{prefix}_denoising_process.png",
        cell_size=72,
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated results: {output_dir}")
