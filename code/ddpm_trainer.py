"""Reusable training loop for unconditional and class-conditional DDPMs."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ddpm_common import (
    count_parameters,
    ensure_dir,
    environment_info,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_json,
    save_tensor_grid,
    set_seed,
    write_csv,
)
from ddpm_data import get_dataset
from ddpm_diffusion import Diffusion, build_diffusion
from ddpm_models import SimpleUNet, build_model


def _plot_loss(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot([row["epoch"] for row in history], [row["loss"] for row in history], marker="o")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("DDPM training loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint_state(
    model: SimpleUNet,
    diffusion: Diffusion,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_loss: float,
    history: List[Dict[str, Any]],
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "model": model.state_dict(),
        "model_config": model.model_config(),
        "diffusion_config": diffusion.config(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "history": history,
        "train_config": train_config,
    }


def load_model_bundle(
    checkpoint_path: Path,
    device_name: str = "auto",
) -> Tuple[SimpleUNet, Diffusion, Dict[str, Any], torch.device]:
    device = get_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = build_diffusion(checkpoint["diffusion_config"], device)
    return model, diffusion, checkpoint, device


def train_diffusion(config: Dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed, bool(config.get("deterministic", False)))
    device = get_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])
    data_dir = ensure_dir(config["data_dir"])
    save_json(config, output_dir / "config.json")
    save_json(environment_info(), output_dir / "environment.json")

    dataset, image_channels, num_classes = get_dataset(
        config["dataset"],
        data_dir,
        train=True,
        max_samples=config.get("max_train_samples"),
        seed=seed,
        download=bool(config.get("download", True)),
        cifar_archive=Path(config["cifar_archive"]) if config.get("cifar_archive") else None,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 0)) > 0,
        generator=loader_generator,
    )
    conditional = bool(config.get("conditional", False))
    model = SimpleUNet(
        image_channels=image_channels,
        base_channels=int(config["base_channels"]),
        num_classes=num_classes if conditional else None,
    ).to(device)
    diffusion = Diffusion(
        timesteps=int(config["timesteps"]),
        beta_schedule=config.get("beta_schedule", "linear"),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        cosine_s=float(config.get("cosine_s", 0.008)),
        device=str(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, global_step, best_loss = 1, 0, math.inf
    history: List[Dict[str, Any]] = []

    resume = config.get("resume")
    if resume:
        resume_path = output_dir / "latest.pt" if resume == "auto" else Path(resume)
        checkpoint = load_checkpoint(resume_path, device)
        if checkpoint["model_config"] != model.model_config():
            raise ValueError("The resume checkpoint model does not match the requested configuration.")
        if checkpoint["diffusion_config"] != diffusion.config():
            raise ValueError("The resume checkpoint diffusion schedule does not match.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", math.inf))
        history = list(checkpoint.get("history", []))

    preview_count = 20 if conditional else int(config.get("preview_images", 16))
    preview_noise = torch.randn(
        preview_count, image_channels, 32, 32, device=device
    )
    preview_labels: Optional[torch.Tensor] = None
    if conditional:
        preview_labels = torch.arange(preview_count, device=device) % num_classes

    print(f"Device: {device}")
    print(f"Dataset: {config['dataset']} ({len(dataset)} samples)")
    print(f"Conditional: {conditional}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Output: {output_dir}")
    training_start = time.perf_counter()

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        seen_samples = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.shape[0]
            timesteps = torch.randint(
                0, diffusion.timesteps, (batch_size,), device=device
            )
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise)
            train_labels: Optional[torch.Tensor] = None
            if conditional:
                train_labels = labels.clone()
                drop_mask = torch.rand(batch_size, device=device) < float(
                    config.get("label_dropout", 0.1)
                )
                train_labels[drop_mask] = int(model.null_label)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                predicted_noise = model(noisy_images, timesteps, train_labels)
                loss = F.mse_loss(predicted_noise, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * batch_size
            seen_samples += batch_size
            global_step += 1
            if batch_index % int(config.get("log_every", 20)) == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch}/{config['epochs']} | batch {batch_index}/{len(train_loader)} "
                    f"| loss {loss_sum / seen_samples:.6f}"
                )

        epoch_loss = loss_sum / max(1, seen_samples)
        elapsed = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "epoch_seconds": elapsed,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        is_best = epoch_loss < best_loss
        best_loss = min(best_loss, epoch_loss)
        state = _checkpoint_state(
            model, diffusion, optimizer, scaler, epoch, global_step, best_loss, history, config
        )
        save_checkpoint(state, output_dir / "latest.pt")
        if is_best:
            save_checkpoint(state, output_dir / "best.pt")
        if epoch % int(config.get("checkpoint_every", 5)) == 0 or epoch == int(config["epochs"]):
            save_checkpoint(state, output_dir / f"epoch_{epoch:03d}.pt")
        write_csv(history, output_dir / "metrics.csv")
        _plot_loss(history, output_dir / "loss_curve.png")

        if int(config.get("sample_every", 1)) > 0 and epoch % int(config.get("sample_every", 1)) == 0:
            samples, _ = diffusion.sample_ddim(
                model,
                shape=preview_noise.shape,
                sampling_steps=min(
                    int(config.get("preview_sampling_steps", 50)), diffusion.timesteps
                ),
                eta=0.0,
                labels=preview_labels,
                guidance_scale=float(config.get("preview_guidance_scale", 2.0)),
                initial_noise=preview_noise,
            )
            save_tensor_grid(
                samples,
                output_dir / f"sample_epoch_{epoch:03d}.png",
                nrow=10 if conditional else 4,
            )

    total_seconds = time.perf_counter() - training_start
    save_json(
        {
            "epochs_completed": int(config["epochs"]),
            "best_loss": best_loss,
            "total_seconds_this_run": total_seconds,
            "parameter_count": count_parameters(model),
            "checkpoint": str(output_dir / "latest.pt"),
        },
        output_dir / "training_summary.json",
    )
    print(f"Training complete in {total_seconds / 60:.2f} min")
    return output_dir / "latest.pt"

