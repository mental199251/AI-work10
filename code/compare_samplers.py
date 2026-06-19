"""Compare ancestral DDPM sampling against accelerated DDIM sampling."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from ddpm_common import ensure_dir, save_labeled_grid, save_sampling_gif, set_seed, write_csv
from ddpm_metrics import evaluate_conditional_samples, load_classifier
from ddpm_trainer import load_model_bundle
from project_cli import project_path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_sample(
    diffusion: Any,
    model: torch.nn.Module,
    sampler: str,
    steps: Optional[int],
    noise: torch.Tensor,
    labels: Optional[torch.Tensor],
    guidance_scale: float,
    return_frames: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    synchronize(noise.device)
    started = time.perf_counter()
    samples, frames = diffusion.sample(
        model,
        shape=noise.shape,
        sampler=sampler,
        sampling_steps=steps,
        labels=labels,
        guidance_scale=guidance_scale,
        initial_noise=noise,
        return_frames=return_frames,
        frame_count=20,
    )
    synchronize(noise.device)
    return samples, frames, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/mnist_conditional/latest.pt")
    parser.add_argument("--classifier", default="outputs/mnist_classifier/classifier.pt")
    parser.add_argument("--output-dir", default="outputs/sampler_comparison")
    parser.add_argument("--ddim-steps", type=int, nargs="+", default=[100, 50, 25])
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(project_path(args.output_dir))
    model, diffusion, checkpoint, device = load_model_bundle(project_path(args.checkpoint), args.device)
    conditional = model.num_classes is not None
    count = args.samples_per_class * 10 if conditional else 20
    channels = int(checkpoint["model_config"]["image_channels"])
    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class) if conditional else None
    noise = torch.randn(count, channels, 32, 32, device=device)
    classifier_path = project_path(args.classifier)
    classifier = load_classifier(classifier_path, device) if conditional and classifier_path.exists() else None

    # Warm up kernels before timing; the result is intentionally discarded.
    diffusion.sample_ddim(
        model,
        shape=(min(2, count), channels, 32, 32),
        sampling_steps=min(5, diffusion.timesteps),
        labels=labels[:2] if labels is not None else None,
        guidance_scale=args.guidance_scale,
    )
    rows: List[Dict[str, Any]] = []
    visual_groups = []
    visual_labels: List[str] = []
    ddpm_samples, ddpm_frames, ddpm_seconds = timed_sample(
        diffusion, model, "ddpm", None, noise, labels, args.guidance_scale, True
    )
    results = [("DDPM", diffusion.timesteps, ddpm_samples, ddpm_seconds)]
    save_sampling_gif(ddpm_frames, output_dir / "ddpm_denoising_process.gif", nrow=5)
    for steps in args.ddim_steps:
        samples, frames, seconds = timed_sample(
            diffusion, model, "ddim", steps, noise, labels, args.guidance_scale, steps == args.ddim_steps[-1]
        )
        results.append(("DDIM", steps, samples, seconds))
        if frames:
            save_sampling_gif(frames, output_dir / "ddim_denoising_process.gif", nrow=5)

    for sampler, steps, samples, seconds in results:
        row: Dict[str, Any] = {
            "sampler": sampler,
            "steps": steps,
            "total_seconds": seconds,
            "seconds_per_image": seconds / count,
            "sample_count": count,
        }
        if classifier is not None and labels is not None:
            metrics, _ = evaluate_conditional_samples(classifier, samples, labels, device)
            row.update(metrics)
        rows.append(row)
        if conditional:
            visual_indices = torch.arange(10) * args.samples_per_class
            visual_groups.append(samples[visual_indices])
            visual_labels.extend(
                [f"{sampler}-{steps}, y={digit}" for digit in range(10)]
            )
        else:
            take = min(10, count)
            visual_groups.append(samples[:take])
            visual_labels.extend([f"{sampler}-{steps}" for _ in range(take)])
    save_labeled_grid(
        torch.cat(visual_groups),
        visual_labels,
        output_dir / "ddpm_vs_ddim_grid.png",
        nrow=min(10, count),
        cell_size=72,
    )
    write_csv(rows, output_dir / "sampler_comparison.csv")

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    names = [f"{row['sampler']}\n{row['steps']} steps" for row in rows]
    axis.bar(names, [row["total_seconds"] for row in rows], color="#4C78A8")
    axis.set_ylabel("Sampling time (seconds)")
    axis.set_title(f"Sampling {count} images")
    figure.tight_layout()
    figure.savefig(output_dir / "sampling_speed_comparison.png", dpi=180)
    plt.close(figure)
    if classifier is not None:
        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        axis.plot(names, [row["conditional_accuracy"] for row in rows], marker="o", label="accuracy")
        axis.plot(names, [row["mean_confidence"] for row in rows], marker="s", label="confidence")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Classifier score")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "sampling_quality_comparison.png", dpi=180)
        plt.close(figure)
    print(f"Sampler comparison: {output_dir}")


if __name__ == "__main__":
    main()
