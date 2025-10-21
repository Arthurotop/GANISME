from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
except ImportError:  # Soft dependency for metrics
    FrechetInceptionDistance = None  # type: ignore[assignment]
    InceptionScore = None  # type: ignore[assignment]


@dataclass
class TrainingHistory:
    generator_loss: List[float]
    discriminator_loss: List[float]
    fid: List[float]
    inception: List[float]

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "generator_loss": self.generator_loss,
            "discriminator_loss": self.discriminator_loss,
            "fid": self.fid,
            "inception": self.inception,
        }


class ArtGANTrainer:
    """
    Unified trainer supporting several GAN variants for artwork generation.

    Modes
    -----
    basic : Small fully connected GAN for quick experiments.
    excellent : WGAN-GP with residual blocks and spectral normalisation.
    dcgan : Convolutional DCGAN aligned with the original architecture.
    """

    def __init__(
        self,
        mode: str = "basic",
        image_size: int = 64,
        channels: int = 3,
        latent_dim: int = 128,
        device: Optional[str] = None,
        fid_every: int = 1,
        fid_samples: int = 512,
        early_stopping_metric: Optional[str] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        early_stopping_mode: Optional[str] = None,
    ) -> None:
        self.mode = mode.lower()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fid_every = max(1, fid_every)
        self.fid_samples = fid_samples
        self.early_stopping_metric = (
            early_stopping_metric.lower() if early_stopping_metric else None
        )
        self.early_stopping_patience = max(1, early_stopping_patience)
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_mode = (
            early_stopping_mode.lower() if early_stopping_mode else None
        )
        self._es_best: Optional[float] = None
        self._es_wait = 0
        self._es_best_epoch = -1

        self._last_generator_optimizer_state: Optional[Dict[str, Any]] = None
        self._last_discriminator_optimizer_state: Optional[Dict[str, Any]] = None

        self.generator: nn.Module
        self.discriminator: nn.Module
        self.criterion: Optional[nn.Module] = None

        self.history = TrainingHistory([], [], [], [])

        self._build_models()
        self._setup_metrics()
        self._validate_early_stopping_config()

    def _build_models(self) -> None:
        if self.mode == "basic":
            self._init_basic_gan()
        elif self.mode == "excellent":
            self._init_excellent_gan()
        elif self.mode == "dcgan":
            self._init_dcgan()
        else:
            raise ValueError("Mode must be one of: basic, excellent, dcgan.")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def _setup_metrics(self) -> None:
        self.fid_metric: Optional[FrechetInceptionDistance]
        self.inception_metric: Optional[InceptionScore]

        if FrechetInceptionDistance and InceptionScore:
            self.fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
            self.inception_metric = InceptionScore(normalize=True).to(self.device)
        else:
            self.fid_metric = None
            self.inception_metric = None

    def _validate_early_stopping_config(self) -> None:
        if not self.early_stopping_metric:
            return

        valid_metrics = {
            "generator_loss",
            "discriminator_loss",
            "fid",
            "inception",
        }

        if self.early_stopping_metric not in valid_metrics:
            raise ValueError(
                f"Unsupported early stopping metric '{self.early_stopping_metric}'. "
                f"Choose from {sorted(valid_metrics)}."
            )

        if self.early_stopping_metric in {"fid", "inception"}:
            if not self.fid_metric or not self.inception_metric or self.fid_samples <= 0:
                raise ValueError(
                    "Early stopping on FID/Inception requires torchmetrics and "
                    "a positive 'fid_samples'."
                )
            if self.fid_every > 1:
                raise ValueError(
                    "Early stopping on FID/Inception requires 'fid_every' to be 1 "
                    "so the metric is computed each epoch."
                )

        if self.early_stopping_mode not in {"min", "max", None}:
            raise ValueError(
                "early_stopping_mode must be 'min', 'max', or None for auto-selection."
            )

        if not self.early_stopping_mode:
            if self.early_stopping_metric in {"generator_loss", "discriminator_loss", "fid"}:
                self.early_stopping_mode = "min"
            else:
                self.early_stopping_mode = "max"

    # region architectures -------------------------------------------------
    def _init_basic_gan(self) -> None:
        flat_dim = self.channels * self.image_size * self.image_size
        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, flat_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def _init_dcgan(self) -> None:
        feature_maps_gen = 64

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_dim, feature_maps_gen * 8, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(feature_maps_gen * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_maps_gen * 8, feature_maps_gen * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps_gen * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_maps_gen * 4, feature_maps_gen * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps_gen * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_maps_gen * 2, feature_maps_gen, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_gen, self.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        feature_maps_disc = 64

        def sn_conv(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
        ) -> nn.Module:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            return nn.utils.spectral_norm(conv)

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.channels, feature_maps_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(feature_maps_disc, feature_maps_disc * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(feature_maps_disc * 2, feature_maps_disc * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(feature_maps_disc * 4, feature_maps_disc * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_maps_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(feature_maps_disc * 8, 1, 4, 1, 0),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def _init_excellent_gan(self) -> None:
        class ResidualBlock(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.relu(x + self.block(x), inplace=True)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        def spectral_conv(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
        ) -> nn.Module:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            return nn.utils.spectral_norm(conv)

        self.discriminator = nn.Sequential(
            spectral_conv(self.channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(512, 1, 4, 1, 0),
        )

        self.criterion = None  # WGAN uses critic scores directly

    # endregion ------------------------------------------------------------

    def train(
        self,
        dataloader: Iterable,
        epochs: int,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        gp_lambda: float = 10.0,
        critic_steps: int = 5,
    ) -> TrainingHistory:
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )

        for epoch in range(epochs):
            running_g_loss = 0.0
            running_d_loss = 0.0
            batches = 0

            for batch in dataloader:
                real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
                real_images = real_images.to(self.device)
                real_images = self._ensure_range(real_images)
                batch_size = real_images.size(0)

                noise = torch.randn(batch_size, self.latent_dim, device=self.device)

                if self.mode == "basic":
                    d_loss, g_loss = self._train_basic_step(
                        real_images, noise, d_optimizer, g_optimizer
                    )
                elif self.mode == "dcgan":
                    d_loss, g_loss = self._train_dcgan_step(
                        real_images, noise, d_optimizer, g_optimizer
                    )
                else:
                    d_loss, g_loss = self._train_wgan_gp_step(
                        real_images,
                        noise,
                        d_optimizer,
                        g_optimizer,
                        gp_lambda,
                        critic_steps,
                    )

                running_d_loss += d_loss
                running_g_loss += g_loss
                batches += 1

            epoch_d_loss = running_d_loss / max(1, batches)
            epoch_g_loss = running_g_loss / max(1, batches)
            self.history.discriminator_loss.append(epoch_d_loss)
            self.history.generator_loss.append(epoch_g_loss)

            if (
                self.fid_metric
                and self.inception_metric
                and self.fid_samples > 0
                and (epoch + 1) % self.fid_every == 0
            ):
                fid_value, inception_value = self._evaluate_metrics(dataloader)
            else:
                fid_value = float("nan")
                inception_value = float("nan")

            self.history.fid.append(fid_value)
            self.history.inception.append(inception_value)

            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"D_loss: {epoch_d_loss:.4f} "
                f"G_loss: {epoch_g_loss:.4f} "
                f"FID: {fid_value:.2f} "
                f"Inception: {inception_value:.3f}"
            )

            if self.early_stopping_metric:
                metric_map = {
                    "generator_loss": epoch_g_loss,
                    "discriminator_loss": epoch_d_loss,
                    "fid": fid_value,
                    "inception": inception_value,
                }
                stop_metric = metric_map[self.early_stopping_metric]
                if self._check_early_stopping(stop_metric, epoch):
                    best_epoch = self._es_best_epoch + 1
                    best_value = self._es_best if self._es_best is not None else stop_metric
                    print(
                        f"Early stopping triggered at epoch {epoch + 1} "
                        f"(best {self.early_stopping_metric}: {best_value:.4f} "
                        f"at epoch {best_epoch})."
                    )
                    break

        self._last_generator_optimizer_state = g_optimizer.state_dict()
        self._last_discriminator_optimizer_state = d_optimizer.state_dict()

        return self.history

    # region training steps ------------------------------------------------
    def _train_basic_step(
        self,
        real_images: torch.Tensor,
        noise: torch.Tensor,
        d_optimizer: torch.optim.Optimizer,
        g_optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        batch_size = real_images.size(0)
        flat_real = real_images.view(batch_size, -1)
        fake_images = self.generator(noise)

        d_optimizer.zero_grad(set_to_none=True)
        real_logits = self.discriminator(flat_real.detach())
        fake_logits = self.discriminator(fake_images.detach())
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_real = self.criterion(real_logits, real_labels)
        loss_fake = self.criterion(fake_logits, fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad(set_to_none=True)
        fake_logits = self.discriminator(fake_images)
        g_loss = self.criterion(fake_logits, real_labels)
        g_loss.backward()
        g_optimizer.step()

        return d_loss.item(), g_loss.item()

    def _train_dcgan_step(
        self,
        real_images: torch.Tensor,
        noise: torch.Tensor,
        d_optimizer: torch.optim.Optimizer,
        g_optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        fake_images = self.generator(noise.view(noise.size(0), self.latent_dim, 1, 1))

        d_optimizer.zero_grad(set_to_none=True)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_real = self.criterion(real_logits, real_labels)
        loss_fake = self.criterion(fake_logits, fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad(set_to_none=True)
        fake_logits = self.discriminator(fake_images)
        g_loss = self.criterion(fake_logits, real_labels)
        g_loss.backward()
        g_optimizer.step()

        return d_loss.item(), g_loss.item()

    def _train_wgan_gp_step(
        self,
        real_images: torch.Tensor,
        noise: torch.Tensor,
        d_optimizer: torch.optim.Optimizer,
        g_optimizer: torch.optim.Optimizer,
        gp_lambda: float,
        critic_steps: int,
    ) -> Tuple[float, float]:
        critic_steps = max(1, critic_steps)

        fake_images = self.generator(noise.view(noise.size(0), self.latent_dim, 1, 1))

        d_optimizer.zero_grad(set_to_none=True)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        wasserstein_distance = fake_logits.mean() - real_logits.mean()
        gradient_penalty = self._gradient_penalty(real_images, fake_images.detach())
        d_loss = wasserstein_distance + gp_lambda * gradient_penalty
        d_loss.backward()
        d_optimizer.step()

        d_loss_value = d_loss.item()

        for _ in range(critic_steps - 1):
            noise = torch.randn(real_images.size(0), self.latent_dim, device=self.device)
            fake_images = self.generator(noise.view(noise.size(0), self.latent_dim, 1, 1))
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = self.discriminator(real_images)
            fake_logits = self.discriminator(fake_images.detach())
            wasserstein_distance = fake_logits.mean() - real_logits.mean()
            gradient_penalty = self._gradient_penalty(real_images, fake_images.detach())
            critic_loss = wasserstein_distance + gp_lambda * gradient_penalty
            critic_loss.backward()
            d_optimizer.step()
            d_loss_value = critic_loss.item()

        g_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(real_images.size(0), self.latent_dim, device=self.device)
        fake_images = self.generator(noise.view(noise.size(0), self.latent_dim, 1, 1))
        fake_logits = self.discriminator(fake_images)
        g_loss = -fake_logits.mean()
        g_loss.backward()
        g_optimizer.step()

        return d_loss_value, g_loss.item()

    # endregion ------------------------------------------------------------

    def _gradient_penalty(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=self.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        disc_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def _evaluate_metrics(self, dataloader: Iterable) -> Tuple[float, float]:
        if not self.fid_metric or not self.inception_metric:
            return float("nan"), float("nan")
        if self.fid_samples <= 0:
            return float("nan"), float("nan")

        self.fid_metric.reset()
        self.inception_metric.reset()

        real_collected = 0
        was_training = self.generator.training
        self.generator.eval()

        for batch in dataloader:
            real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_images = real_images.to(self.device)
            real_images = self._ensure_range(real_images)
            batch_size = real_images.size(0)

            fake_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            if self.mode == "basic":
                fake_images = self.generator(fake_noise).view_as(real_images)
            else:
                fake_images = self.generator(
                    fake_noise.view(fake_noise.size(0), self.latent_dim, 1, 1)
                )

            real_ready = torch.clamp((real_images + 1) / 2, 0.0, 1.0)
            fake_ready = torch.clamp((fake_images + 1) / 2, 0.0, 1.0)

            self.fid_metric.update(real_ready, real=True)
            self.fid_metric.update(fake_ready, real=False)
            self.inception_metric.update(fake_ready)

            real_collected += batch_size
            if real_collected >= self.fid_samples:
                break

        if was_training:
            self.generator.train()

        fid_value = self.fid_metric.compute().item()
        inception_mean, _ = self.inception_metric.compute()
        return float(fid_value), float(inception_mean.item())

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.generate(num_samples=num_samples, return_on_cpu=False)

    def _ensure_range(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.min() >= 0.0 and tensor.max() <= 1.0:
            return tensor * 2 - 1
        return tensor.clamp(-1, 1)

    def _check_early_stopping(self, metric_value: float, epoch: int) -> bool:
        if math.isnan(metric_value):
            return False

        if self._es_best is None:
            self._es_best = metric_value
            self._es_best_epoch = epoch
            self._es_wait = 0
            return False

        mode = self.early_stopping_mode or "min"
        delta = self.early_stopping_min_delta

        improved = (
            metric_value < (self._es_best - delta)
            if mode == "min"
            else metric_value > (self._es_best + delta)
        )

        if improved:
            self._es_best = metric_value
            self._es_best_epoch = epoch
            self._es_wait = 0
            return False

        self._es_wait += 1
        return self._es_wait >= self.early_stopping_patience

    def _sample_generator_noise(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.mode == "basic":
            return torch.randn(batch_size, self.latent_dim, device=self.device)
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

    def _generate_from_noise(self, noise: torch.Tensor) -> torch.Tensor:
        batch_size = noise.size(0)
        noise = noise.to(self.device)

        if self.mode == "basic":
            if noise.dim() != 2 or noise.size(1) != self.latent_dim:
                raise ValueError(
                    f"Expected noise of shape (batch, {self.latent_dim}) for basic mode; "
                    f"got {tuple(noise.shape)}."
                )
            generated = self.generator(noise)
            return generated.view(batch_size, self.channels, self.image_size, self.image_size)

        if noise.dim() == 2:
            if noise.size(1) != self.latent_dim:
                raise ValueError(
                    f"Noise second dimension must equal latent_dim ({self.latent_dim})."
                )
            noise = noise.view(batch_size, self.latent_dim, 1, 1)
        elif noise.dim() == 4:
            expected = (self.latent_dim, 1, 1)
            if noise.shape[1:] != expected:
                raise ValueError(
                    f"Expected noise of shape (batch, {expected[0]}, {expected[1]}, {expected[2]}). "
                    f"Got {tuple(noise.shape)}."
                )
        else:
            raise ValueError(
                "Noise must be either 2-D (batch, latent_dim) or 4-D "
                "(batch, latent_dim, 1, 1) for convolutional modes."
            )

        return self.generator(noise)

    def save_checkpoint(self, path: str, include_optimizers: bool = False) -> None:
        checkpoint = {
            "mode": self.mode,
            "image_size": self.image_size,
            "channels": self.channels,
            "latent_dim": self.latent_dim,
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "history": self.history.as_dict(),
            "early_stopping": {
                "metric": self.early_stopping_metric,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "mode": self.early_stopping_mode,
                "best": self._es_best,
                "best_epoch": self._es_best_epoch,
                "wait": self._es_wait,
            },
            "fid_every": self.fid_every,
            "fid_samples": self.fid_samples,
        }

        if include_optimizers:
            if (
                self._last_generator_optimizer_state is None
                or self._last_discriminator_optimizer_state is None
            ):
                raise RuntimeError(
                    "Optimizer states requested for saving but unavailable. "
                    "Train the model or set `include_optimizers=False`."
                )

            checkpoint["optimizers"] = {
                "generator": self._last_generator_optimizer_state,
                "discriminator": self._last_discriminator_optimizer_state,
            }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._load_checkpoint_dict(checkpoint, strict=strict)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None,
        strict: bool = True,
    ) -> "ArtGANTrainer":
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            map_location = torch.device(resolved_device)
        except (RuntimeError, ValueError):
            map_location = torch.device("cpu")
            resolved_device = "cpu"

        checkpoint = torch.load(path, map_location=map_location)
        trainer = cls(
            mode=checkpoint.get("mode", "basic"),
            image_size=checkpoint.get("image_size", 64),
            channels=checkpoint.get("channels", 3),
            latent_dim=checkpoint.get("latent_dim", 128),
            device=resolved_device,
            fid_every=checkpoint.get("fid_every", 1),
            fid_samples=checkpoint.get("fid_samples", 512),
        )
        trainer._load_checkpoint_dict(checkpoint, strict=strict)
        return trainer

    def _load_checkpoint_dict(self, checkpoint: Dict[str, Any], strict: bool) -> None:
        required_keys = {
            "mode",
            "image_size",
            "channels",
            "latent_dim",
            "generator_state",
            "discriminator_state",
            "history",
        }
        missing = required_keys.difference(checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint missing keys: {missing}")

        if (
            checkpoint["mode"] != self.mode
            or checkpoint["image_size"] != self.image_size
            or checkpoint["channels"] != self.channels
            or checkpoint["latent_dim"] != self.latent_dim
        ):
            raise ValueError(
                "Checkpoint configuration mismatch. Instantiate ArtGANTrainer with the same "
                "mode/image_size/channels/latent_dim before loading."
            )

        self.generator.load_state_dict(checkpoint["generator_state"], strict=strict)
        self.discriminator.load_state_dict(checkpoint["discriminator_state"], strict=strict)

        history_dict = checkpoint.get("history", {})
        self.history = TrainingHistory(
            generator_loss=history_dict.get("generator_loss", []),
            discriminator_loss=history_dict.get("discriminator_loss", []),
            fid=history_dict.get("fid", []),
            inception=history_dict.get("inception", []),
        )

        es_state = checkpoint.get("early_stopping", {})
        if es_state:
            self.early_stopping_metric = es_state.get("metric", self.early_stopping_metric)
            self.early_stopping_patience = es_state.get("patience", self.early_stopping_patience)
            self.early_stopping_min_delta = es_state.get("min_delta", self.early_stopping_min_delta)
            self.early_stopping_mode = es_state.get("mode", self.early_stopping_mode)
            self._es_best = es_state.get("best")
            self._es_best_epoch = es_state.get("best_epoch", -1)
            self._es_wait = es_state.get("wait", 0)

        optim_state = checkpoint.get("optimizers")
        if isinstance(optim_state, dict):
            self._last_generator_optimizer_state = optim_state.get("generator")
            self._last_discriminator_optimizer_state = optim_state.get("discriminator")

    def generate(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        denormalize: bool = False,
        return_on_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Create new images from the trained generator.

        Parameters
        ----------
        num_samples: int
            Number of images to produce when `noise` is not provided.
        batch_size: Optional[int]
            Batch size to use during generation. Defaults to min(num_samples, 64).
        noise: Optional[torch.Tensor]
            Optional latent noise tensor. When provided, its first dimension defines
            the number of samples, making `num_samples` ignored.
        denormalize: bool
            Whether to map outputs from [-1, 1] back to [0, 1].
        return_on_cpu: bool
            If True (default) the returned tensor resides on CPU memory.

        Returns
        -------
        torch.Tensor
            Generated images with shape (N, C, H, W).
        """
        if noise is not None:
            total_samples = noise.size(0)
            if total_samples <= 0:
                raise ValueError("Provided noise tensor must contain at least one sample.")
            num_samples = total_samples
        else:
            if num_samples <= 0:
                raise ValueError("num_samples must be a positive integer when noise is not provided.")

        batch_size = batch_size or min(num_samples, 64)
        if batch_size <= 0:
            raise ValueError("batch_size must resolve to a positive integer.")

        outputs: List[torch.Tensor] = []
        was_training = self.generator.training
        self.generator.eval()

        try:
            with torch.no_grad():
                produced = 0
                while produced < num_samples:
                    current_batch = min(batch_size, num_samples - produced)

                    if noise is not None:
                        noise_batch = noise[produced : produced + current_batch]
                    else:
                        noise_batch = self._sample_generator_noise(current_batch)

                    fake_images = self._generate_from_noise(noise_batch).detach()

                    if denormalize:
                        fake_images = fake_images.add(1).div(2).clamp(0, 1)

                    if return_on_cpu:
                        outputs.append(fake_images.cpu())
                    else:
                        outputs.append(fake_images)

                    produced += current_batch
        finally:
            if was_training:
                self.generator.train()

        return torch.cat(outputs, dim=0)

    def plot_history(self) -> None:
        metrics = self.history.as_dict()
        epochs = range(1, len(metrics["generator_loss"]) + 1)

        if not epochs:
            raise ValueError("No training history available to plot.")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(epochs, metrics["generator_loss"], label="Generator")
        axes[0].plot(epochs, metrics["discriminator_loss"], label="Discriminator")
        axes[0].set_title("Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, metrics["fid"], label="FID")
        axes[1].set_title("FID (lower is better)")
        axes[1].set_xlabel("Epoch")

        axes[2].plot(epochs, metrics["inception"], label="Inception Score")
        axes[2].set_title("Inception Score")
        axes[2].set_xlabel("Epoch")

        plt.tight_layout()
        plt.show()
