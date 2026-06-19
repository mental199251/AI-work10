from pathlib import Path

import torch
from torchvision import datasets, transforms

from ddpm_common import (
    ensure_dir,
    get_device,
    load_model_from_checkpoint,
    save_labeled_sequence,
    save_sampling_gif,
    save_tensor_grid,
    set_seed,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT = PROJECT_DIR / "outputs" / "mnist" / "mnist_ddpm_latest.pt"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "mnist_results"

NUM_IMAGES = 16
GIF_EVERY = 25
DOWNLOAD_DATASET = True
SKIP_FORWARD_DEMO = False
DEVICE = "auto"
SEED = 123


def save_forward_noising_demo(schedule, data_dir, output_dir, download_dataset, device):
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download_dataset,
        transform=transform,
    )
    image, label = dataset[0]
    image = image.unsqueeze(0).to(device)
    fixed_noise = torch.randn_like(image)
    steps = [
        0,
        schedule.timesteps // 5,
        schedule.timesteps * 2 // 5,
        schedule.timesteps * 3 // 5,
        schedule.timesteps * 4 // 5,
        schedule.timesteps - 1,
    ]

    tensors = []
    labels = []
    for step in steps:
        t = torch.tensor([step], device=device, dtype=torch.long)
        tensors.append(schedule.q_sample(image, t, fixed_noise).cpu())
        labels.append(f"t={step}")

    output_path = output_dir / f"mnist_forward_noise_label_{label}.png"
    save_labeled_sequence(tensors, labels, output_path)
    return output_path


def main():
    set_seed(SEED)
    device = get_device(DEVICE)
    output_dir = ensure_dir(OUTPUT_DIR)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}. Please run train_mnist_ddpm.py first.")

    model, schedule, checkpoint = load_model_from_checkpoint(CHECKPOINT, device)
    channels = checkpoint["config"]["image_channels"]
    image_size = checkpoint["config"]["image_size"]

    forward_demo_path = None
    if not SKIP_FORWARD_DEMO:
        forward_demo_path = save_forward_noising_demo(schedule, DATA_DIR, output_dir, DOWNLOAD_DATASET, device)

    samples, frames = schedule.sample(
        model,
        shape=(NUM_IMAGES, channels, image_size, image_size),
        save_every=GIF_EVERY,
        return_frames=True,
    )

    nrow = max(1, int(NUM_IMAGES ** 0.5))
    generated_grid = output_dir / "mnist_generated_grid.png"
    denoising_gif = output_dir / "mnist_denoising_process.gif"
    save_tensor_grid(samples, generated_grid, nrow=nrow)
    save_sampling_gif(frames, denoising_gif, nrow=nrow)

    print("Required task: show MNIST DDPM results")
    print(f"Generated grid: {generated_grid}")
    print(f"Denoising GIF: {denoising_gif}")
    if forward_demo_path is not None:
        print(f"Forward noising demo: {forward_demo_path}")


if __name__ == "__main__":
    main()
