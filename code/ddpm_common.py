"""Shared utilities for the DDPM experiments."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image


def ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.detach().clamp(-1, 1) + 1.0) * 0.5


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    image = (image * 255).round().byte()
    if image.shape[0] == 1:
        return Image.fromarray(image[0].numpy(), mode="L").convert("RGB")
    return Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB")


def save_tensor_grid(images: torch.Tensor, path: os.PathLike, nrow: int = 4) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    save_image(unnormalize(images).cpu(), path, nrow=nrow)
    return path


def save_labeled_grid(
    images: torch.Tensor,
    labels: Sequence[str],
    path: os.PathLike,
    nrow: int,
    cell_size: int = 80,
) -> Path:
    """Save a grid with one short label below every image."""
    path = Path(path)
    ensure_dir(path.parent)
    images = unnormalize(images).cpu()
    if len(images) != len(labels):
        raise ValueError("The number of labels must match the number of images.")
    ncol = nrow
    nrows = (len(images) + ncol - 1) // ncol
    canvas = Image.new("RGB", (ncol * cell_size, nrows * (cell_size + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, ncol)
        image_pil = tensor_to_pil(image).resize((cell_size, cell_size), Image.Resampling.NEAREST)
        x, y = col * cell_size, row * (cell_size + 20)
        canvas.paste(image_pil, (x, y))
        draw.text((x + 3, y + cell_size + 3), str(label), fill="black")
    canvas.save(path)
    return path


def save_labeled_sequence(
    images: Sequence[torch.Tensor],
    labels: Sequence[str],
    path: os.PathLike,
    cell_size: int = 96,
) -> Path:
    batch = torch.cat([image[:1].detach().cpu() for image in images], dim=0)
    return save_labeled_grid(batch, labels, path, nrow=len(labels), cell_size=cell_size)


def save_sampling_gif(
    frames: Sequence[torch.Tensor],
    path: os.PathLike,
    nrow: int = 4,
    duration: int = 180,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    pil_frames: List[Image.Image] = []
    for frame in frames:
        grid = make_grid(unnormalize(frame).cpu(), nrow=nrow)
        if grid.shape[0] == 1:
            grid = grid.repeat(3, 1, 1)
        pil_frames.append(tensor_to_pil(grid))
    if not pil_frames:
        raise ValueError("No frames were provided for GIF generation.")
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    return path


def save_json(data: Dict[str, Any], path: os.PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def load_json(path: os.PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(rows: Iterable[Dict[str, Any]], path: os.PathLike) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def environment_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(state: Dict[str, Any], path: os.PathLike) -> Path:
    """Write through a temporary file so interrupted jobs do not corrupt latest.pt."""
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: os.PathLike, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location=device)
