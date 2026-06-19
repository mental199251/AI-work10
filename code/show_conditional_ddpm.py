"""Visualize class-conditional generation and classifier-free guidance strength."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from ddpm_common import ensure_dir, save_json, save_labeled_grid, save_tensor_grid, set_seed, write_csv
from ddpm_metrics import evaluate_conditional_samples, load_classifier, save_confusion_matrix
from ddpm_trainer import load_model_bundle
from project_cli import project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/mnist_conditional/latest.pt")
    parser.add_argument("--classifier", default="outputs/mnist_classifier/classifier.pt")
    parser.add_argument("--output-dir", default="outputs/conditional_results")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument("--selected-scale", type=float, default=2.0)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(project_path(args.output_dir))
    model, diffusion, checkpoint, device = load_model_bundle(project_path(args.checkpoint), args.device)
    if model.num_classes is None:
        raise ValueError("The checkpoint is not class conditional.")
    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
    noise = torch.randn(len(labels), 1, 32, 32, device=device)
    classifier_path = project_path(args.classifier)
    classifier = load_classifier(classifier_path, device) if classifier_path.exists() else None

    rows: List[Dict[str, float]] = []
    generated_by_scale: Dict[float, torch.Tensor] = {}
    selected_confusion = None
    for scale in args.guidance_scales:
        samples, _ = diffusion.sample_ddim(
            model,
            shape=noise.shape,
            sampling_steps=args.sampling_steps,
            eta=0.0,
            labels=labels,
            guidance_scale=scale,
            initial_noise=noise,
        )
        generated_by_scale[scale] = samples
        row: Dict[str, float] = {"guidance_scale": float(scale)}
        if classifier is not None:
            metrics, confusion = evaluate_conditional_samples(classifier, samples, labels, device)
            row.update(metrics)
            if scale == args.selected_scale:
                selected_confusion = confusion
        rows.append(row)

    selected = generated_by_scale.get(args.selected_scale)
    if selected is None:
        selected = generated_by_scale[args.guidance_scales[0]]
    save_tensor_grid(
        selected,
        output_dir / "conditional_digits_grid.png",
        nrow=args.samples_per_class,
    )

    comparison_images = []
    comparison_labels = []
    for scale in args.guidance_scales:
        images = generated_by_scale[scale]
        # Use one sample per digit so rows compare guidance strength directly.
        indices = torch.arange(10) * args.samples_per_class
        comparison_images.append(images[indices])
        comparison_labels.extend([f"w={scale:g}, y={digit}" for digit in range(10)])
    save_labeled_grid(
        torch.cat(comparison_images),
        comparison_labels,
        output_dir / "guidance_scale_comparison.png",
        nrow=10,
        cell_size=72,
    )
    write_csv(rows, output_dir / "conditional_metrics.csv")
    if classifier is not None:
        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        axis.plot(
            [row["guidance_scale"] for row in rows],
            [row["conditional_accuracy"] for row in rows],
            marker="o",
            label="class accuracy",
        )
        axis.plot(
            [row["guidance_scale"] for row in rows],
            [row["mean_confidence"] for row in rows],
            marker="s",
            label="mean confidence",
        )
        axis.set_xlabel("Guidance scale")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "guidance_accuracy_curve.png", dpi=180)
        plt.close(figure)
        if selected_confusion is not None:
            save_confusion_matrix(
                selected_confusion,
                output_dir / "conditional_confusion_matrix.png",
                f"Conditional generation (guidance={args.selected_scale:g})",
            )
    save_json(
        {
            "checkpoint": str(project_path(args.checkpoint)),
            "sampling_steps": args.sampling_steps,
            "samples_per_class": args.samples_per_class,
            "guidance_scales": args.guidance_scales,
        },
        output_dir / "generation_config.json",
    )
    print(f"Conditional results: {output_dir}")


if __name__ == "__main__":
    main()

