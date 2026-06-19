"""Forward diffusion, ancestral DDPM sampling, and accelerated DDIM sampling."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alpha_bars = torch.cos(
            ((steps / timesteps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5
        ).pow(2)
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        return betas.clamp(1e-4, 0.999).float()
    raise ValueError(f"Unknown beta schedule: {schedule}")


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class Diffusion:
    def __init__(
        self,
        timesteps: int = 300,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: str = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cosine_s = float(cosine_s)
        self.device = torch.device(device)
        betas = make_beta_schedule(
            beta_schedule, self.timesteps, beta_start, beta_end, cosine_s
        ).to(self.device)
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
        self.posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

    def config(self) -> Dict[str, Any]:
        return {
            "timesteps": self.timesteps,
            "beta_schedule": self.beta_schedule,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "cosine_s": self.cosine_s,
        }

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            extract(self.sqrt_alpha_bars, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape) * noise
        )

    def predict_noise(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        conditional = getattr(model, "num_classes", None) is not None
        if not conditional or labels is None:
            return model(x, timesteps, labels)
        null_labels = torch.full_like(labels, int(model.null_label))
        if guidance_scale == 0.0:
            return model(x, timesteps, null_labels)
        # A single doubled batch is faster than two separate forward passes on a GPU.
        model_input = torch.cat((x, x), dim=0)
        time_input = torch.cat((timesteps, timesteps), dim=0)
        label_input = torch.cat((null_labels, labels), dim=0)
        noise_unconditional, noise_conditional = model(
            model_input, time_input, label_input
        ).chunk(2)
        return noise_unconditional + guidance_scale * (
            noise_conditional - noise_unconditional
        )

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        predicted_noise = self.predict_noise(
            model, x, timesteps, labels, guidance_scale
        )
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alpha_bars, timesteps, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)
        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_t
        )
        variance_t = extract(self.posterior_variance, timesteps, x.shape)
        mask = (timesteps != 0).float().reshape(
            x.shape[0], *((1,) * (x.ndim - 1))
        )
        return model_mean + mask * torch.sqrt(variance_t.clamp(min=1e-20)) * torch.randn_like(x)

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: torch.nn.Module,
        shape: Sequence[int],
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        initial_noise: Optional[torch.Tensor] = None,
        return_frames: bool = False,
        frame_count: int = 20,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        model.eval()
        x = torch.randn(tuple(shape), device=self.device) if initial_noise is None else initial_noise.to(self.device).clone()
        labels = labels.to(self.device) if labels is not None else None
        frames: List[torch.Tensor] = []
        if return_frames:
            frames.append(x.detach().cpu())
        frame_interval = max(1, self.timesteps // max(1, frame_count - 1))
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, labels, guidance_scale)
            if return_frames and (
                step == self.timesteps - 1 or step == 0 or step % frame_interval == 0
            ):
                frames.append(x.detach().cpu())
        return x.detach().cpu(), frames

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        shape: Sequence[int],
        sampling_steps: int = 50,
        eta: float = 0.0,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        initial_noise: Optional[torch.Tensor] = None,
        return_frames: bool = False,
        frame_count: int = 20,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps must be between 1 and timesteps")
        model.eval()
        x = torch.randn(tuple(shape), device=self.device) if initial_noise is None else initial_noise.to(self.device).clone()
        labels = labels.to(self.device) if labels is not None else None
        times = torch.linspace(
            self.timesteps - 1, 0, sampling_steps, device=self.device
        ).round().long()
        times = torch.unique_consecutive(times)
        frames: List[torch.Tensor] = []
        if return_frames:
            frames.append(x.detach().cpu())
        frame_interval = max(1, len(times) // max(1, frame_count - 1))
        for index, time_value in enumerate(times):
            step = int(time_value.item())
            previous_step = int(times[index + 1].item()) if index + 1 < len(times) else -1
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            predicted_noise = self.predict_noise(
                model, x, t, labels, guidance_scale
            )
            alpha_bar = self.alpha_bars[step]
            alpha_bar_previous = (
                self.alpha_bars[previous_step]
                if previous_step >= 0
                else torch.tensor(1.0, device=self.device)
            )
            predicted_x0 = (
                x - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            if clip_denoised:
                predicted_x0 = predicted_x0.clamp(-1, 1)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_bar_previous) / (1.0 - alpha_bar))
                * (1.0 - alpha_bar / alpha_bar_previous)
            ).clamp(min=0)
            direction = torch.sqrt(
                (1.0 - alpha_bar_previous - sigma.square()).clamp(min=0)
            ) * predicted_noise
            x = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + direction
                + sigma * torch.randn_like(x)
            )
            if return_frames and (
                index == 0 or index == len(times) - 1 or index % frame_interval == 0
            ):
                frames.append(x.detach().cpu())
        return x.detach().cpu(), frames

    def sample(
        self,
        model: torch.nn.Module,
        shape: Sequence[int],
        sampler: str = "ddpm",
        sampling_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if sampler == "ddpm":
            return self.sample_ddpm(model, shape, **kwargs)
        if sampler == "ddim":
            return self.sample_ddim(
                model,
                shape,
                sampling_steps=sampling_steps or min(50, self.timesteps),
                **kwargs,
            )
        raise ValueError(f"Unknown sampler: {sampler}")


def build_diffusion(config: Dict[str, Any], device: torch.device) -> Diffusion:
    return Diffusion(
        timesteps=int(config["timesteps"]),
        beta_schedule=config.get("beta_schedule", "linear"),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        cosine_s=float(config.get("cosine_s", 0.008)),
        device=str(device),
    )
