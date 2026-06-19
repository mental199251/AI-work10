"""Compare independently trained linear and cosine noise schedules."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch

from ddpm_common import ensure_dir, save_labeled_grid, set_seed, write_csv
from ddpm_metrics import evaluate_conditional_samples, load_classifier
from ddpm_trainer import load_model_bundle
from project_cli import project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-checkpoint", required=True)
    parser.add_argument("--cosine-checkpoint", required=True)
    parser.add_argument("--classifier", default="outputs/mnist_classifier/classifier.pt")
    parser.add_argument("--output-dir", default="outputs/schedule_comparison")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(project_path(args.output_dir))
    linear_model, linear_diffusion, linear_ckpt, device = load_model_bundle(
        project_path(args.linear_checkpoint), args.device
    )
    cosine_model, cosine_diffusion, cosine_ckpt, _ = load_model_bundle(
        project_path(args.cosine_checkpoint), str(device)
    )
    if linear_ckpt["model_config"] != cosine_ckpt["model_config"]:
        raise ValueError("Schedule comparison requires identical model configurations.")
    conditional = linear_model.num_classes is not None
    count = args.samples_per_class * 10 if conditional else 20
    channels = int(linear_ckpt["model_config"]["image_channels"])
    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class) if conditional else None
    noise = torch.randn(count, channels, 32, 32, device=device)
    classifier_path = project_path(args.classifier)
    classifier = load_classifier(classifier_path, device) if conditional and classifier_path.exists() else None
    rows: List[Dict[str, Any]] = []
    visual_groups = []
    visual_labels: List[str] = []
    for name, model, diffusion, checkpoint in (
        ("linear", linear_model, linear_diffusion, linear_ckpt),
        ("cosine", cosine_model, cosine_diffusion, cosine_ckpt),
    ):
        samples, _ = diffusion.sample_ddim(
            model,
            shape=noise.shape,
            sampling_steps=min(args.sampling_steps, diffusion.timesteps),
            labels=labels,
            guidance_scale=args.guidance_scale,
            initial_noise=noise,
        )
        row: Dict[str, Any] = {
            "schedule": name,
            "final_train_loss": checkpoint.get("history", [{}])[-1].get("loss"),
            "best_train_loss": checkpoint.get("best_loss"),
        }
        if classifier is not None and labels is not None:
            metrics, _ = evaluate_conditional_samples(classifier, samples, labels, device)
            row.update(metrics)
        rows.append(row)
        if conditional:
            visual_indices = torch.arange(10) * args.samples_per_class
            visual_groups.append(samples[visual_indices])
            visual_labels.extend([f"{name}, y={digit}" for digit in range(10)])
        else:
            take = min(10, count)
            visual_groups.append(samples[:take])
            visual_labels.extend([name for _ in range(take)])
    write_csv(rows, output_dir / "schedule_comparison.csv")
    save_labeled_grid(
        torch.cat(visual_groups),
        visual_labels,
        output_dir / "linear_vs_cosine_samples.png",
        nrow=min(10, count),
        cell_size=72,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, diffusion in (("linear", linear_diffusion), ("cosine", cosine_diffusion)):
        axes[0].plot(diffusion.betas.cpu(), label=name)
        axes[1].plot(diffusion.alpha_bars.cpu(), label=name)
    axes[0].set_title("Beta schedule")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("beta")
    axes[1].set_title("Cumulative alpha")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("alpha_bar")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "schedule_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for name, checkpoint in (("linear", linear_ckpt), ("cosine", cosine_ckpt)):
        history = checkpoint.get("history", [])
        axis.plot([row["epoch"] for row in history], [row["loss"] for row in history], marker="o", label=name)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "linear_vs_cosine_loss.png", dpi=180)
    plt.close(figure)
    print(f"Schedule comparison: {output_dir}")


if __name__ == "__main__":
    main()
