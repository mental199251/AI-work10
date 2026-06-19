import math
import random
import tarfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unnormalize(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def save_tensor_grid(tensor, path, nrow=4):
    path = Path(path)
    ensure_dir(path.parent)
    save_image(unnormalize(tensor), path, nrow=nrow)


def save_labeled_sequence(tensors, labels, path, cell_size=96):
    path = Path(path)
    ensure_dir(path.parent)
    imgs = []
    for tensor, label in zip(tensors, labels):
        img = tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        img = unnormalize(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        pil = to_pil_image(img)
        pil = pil.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (cell_size, cell_size + 22), "white")
        canvas.paste(pil, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, cell_size + 4), str(label), fill=(0, 0, 0))
        imgs.append(canvas)

    out = Image.new("RGB", (cell_size * len(imgs), cell_size + 22), "white")
    for i, img in enumerate(imgs):
        out.paste(img, (i * cell_size, 0))
    out.save(path)


def to_pil_image(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255).byte()
    if tensor.shape[0] == 1:
        return Image.fromarray(tensor[0].numpy(), mode="L").convert("RGB")
    return Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB")


def save_sampling_gif(frames, path, nrow=4, duration=180):
    path = Path(path)
    ensure_dir(path.parent)
    pil_frames = []
    for frame in frames:
        grid = make_grid(unnormalize(frame.detach().cpu()), nrow=nrow)
        if grid.shape[0] == 1:
            grid = grid.repeat(3, 1, 1)
        pil_frames.append(to_pil_image(grid))

    if not pil_frames:
        raise ValueError("No frames were provided for GIF generation.")

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        groups = min(8, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=32, time_dim=None):
        super().__init__()
        time_dim = time_dim or base_channels * 4

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.init_conv = nn.Conv2d(image_channels, c1, 3, padding=1)

        self.down1 = ResidualBlock(c1, c1, time_dim)
        self.downsample1 = Downsample(c1)
        self.down2 = ResidualBlock(c1, c2, time_dim)
        self.downsample2 = Downsample(c2)
        self.down3 = ResidualBlock(c2, c3, time_dim)
        self.downsample3 = Downsample(c3)

        self.mid1 = ResidualBlock(c3, c3, time_dim)
        self.mid2 = ResidualBlock(c3, c3, time_dim)

        self.upsample3 = Upsample(c3)
        self.up3 = ResidualBlock(c3 + c3, c2, time_dim)
        self.upsample2 = Upsample(c2)
        self.up2 = ResidualBlock(c2 + c2, c1, time_dim)
        self.upsample1 = Upsample(c1)
        self.up1 = ResidualBlock(c1 + c1, c1, time_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(min(8, c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, image_channels, 1),
        )

    def forward(self, x, time):
        time_emb = self.time_mlp(time)

        x = self.init_conv(x)
        skip1 = self.down1(x, time_emb)
        x = self.downsample1(skip1)

        skip2 = self.down2(x, time_emb)
        x = self.downsample2(skip2)

        skip3 = self.down3(x, time_emb)
        x = self.downsample3(skip3)

        x = self.mid1(x, time_emb)
        x = self.mid2(x, time_emb)

        x = self.upsample3(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.up3(x, time_emb)

        x = self.upsample2(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.up2(x, time_emb)

        x = self.upsample1(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.up1(x, time_emb)

        return self.final(x)


def extract(values, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    out = values.gather(-1, timesteps)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionSchedule:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, model, x, timesteps):
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alpha_bars, timesteps, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)

        predicted_noise = model(x, timesteps)
        model_mean = sqrt_recip_alpha_t * (x - betas_t * predicted_noise / sqrt_one_minus_t)

        noise = torch.randn_like(x)
        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)
        nonzero_mask = (timesteps != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t.clamp(min=1e-20)) * noise

    @torch.no_grad()
    def sample(self, model, shape, save_every=25, return_frames=False):
        model.eval()
        x = torch.randn(shape, device=self.device)
        frames = []

        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if return_frames and (step % save_every == 0 or step == self.timesteps - 1):
                frames.append(x.detach().cpu())

        if return_frames and (len(frames) == 0 or not torch.equal(frames[-1], x.detach().cpu())):
            frames.append(x.detach().cpu())
        return x.detach().cpu(), frames


def save_checkpoint(path, model, optimizer, epoch, global_step, config):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        path,
    )


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SimpleUNet(
        image_channels=config["image_channels"],
        base_channels=config["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(
        timesteps=config["timesteps"],
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
        device=device,
    )
    return model, schedule, checkpoint


def subset_dataset(dataset, max_samples):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    indices = list(range(max_samples))
    return torch.utils.data.Subset(dataset, indices)


def prepare_cifar10_data(data_dir):
    data_dir = ensure_dir(data_dir)
    extracted_dir = data_dir / "cifar-10-batches-py"
    archive_path = data_dir / "cifar-10-python.tar.gz"

    if extracted_dir.exists():
        print(f"Using local CIFAR-10 data: {extracted_dir}")
        return

    if archive_path.exists():
        print(f"Extracting local CIFAR-10 archive: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                target = (data_dir / member.name).resolve()
                data_root = data_dir.resolve()
                try:
                    target.relative_to(data_root)
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe path found in archive: {member.name}") from exc
            tar.extractall(data_dir)
        print(f"Extracted CIFAR-10 data to: {extracted_dir}")
        return

    raise FileNotFoundError(
        "CIFAR-10 data was not found. Put cifar-10-python.tar.gz in the data folder, "
        "or put the extracted cifar-10-batches-py folder in the data folder."
    )
