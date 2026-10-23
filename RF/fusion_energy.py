"""Direct differentiable fusion energy for RF-aligned guidance.

All public tensors are BCHW images in the fixed [0, 1] domain.  The energy
contains only source-derived image-fusion constraints; it does not call the
released EM/posterior operator.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _luminance(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError("Expected a BCHW image with one or three channels")
    if image.shape[1] == 1:
        return image
    return (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )


def _sobel_xy(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = image.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    return (
        F.conv2d(image, kernel_x, padding=1),
        F.conv2d(image, kernel_y, padding=1),
    )


def saliency_weights(
    visible_y: torch.Tensor,
    infrared_y: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed soft source-selection weights derived from local edge strength."""
    if temperature <= 0.0:
        raise ValueError("saliency temperature must be positive")
    vx, vy = _sobel_xy(visible_y)
    ix, iy = _sobel_xy(infrared_y)
    visible_strength = (vx.square() + vy.square() + 1e-12).sqrt()
    infrared_strength = (ix.square() + iy.square() + 1e-12).sqrt()
    visible_weight = torch.sigmoid(
        (visible_strength - infrared_strength) / float(temperature)
    ).detach()
    return visible_weight, 1.0 - visible_weight


def _intensity_energy(
    fused_y: torch.Tensor,
    visible_y: torch.Tensor,
    infrared_y: torch.Tensor,
) -> torch.Tensor:
    target = torch.maximum(visible_y, infrared_y)
    return (fused_y - target).abs().flatten(1).mean(1)


def _gradient_energy(
    fused_y: torch.Tensor,
    visible_y: torch.Tensor,
    infrared_y: torch.Tensor,
    scales=(1.0, 0.5, 0.25),
) -> torch.Tensor:
    total = fused_y.new_zeros((fused_y.shape[0],))
    for scale in scales:
        if scale == 1.0:
            fused_s, visible_s, infrared_s = fused_y, visible_y, infrared_y
        else:
            kwargs = {
                "scale_factor": scale,
                "mode": "bilinear",
                "align_corners": False,
                "recompute_scale_factor": False,
            }
            fused_s = F.interpolate(fused_y, **kwargs)
            visible_s = F.interpolate(visible_y, **kwargs)
            infrared_s = F.interpolate(infrared_y, **kwargs)
        fused_x, fused_y_axis = _sobel_xy(fused_s)
        visible_x, visible_y_axis = _sobel_xy(visible_s)
        infrared_x, infrared_y_axis = _sobel_xy(infrared_s)
        target_x = torch.where(
            visible_x.abs() >= infrared_x.abs(), visible_x, infrared_x
        )
        target_y = torch.where(
            visible_y_axis.abs() >= infrared_y_axis.abs(),
            visible_y_axis,
            infrared_y_axis,
        )
        total = total + (
            (fused_x - target_x).abs().flatten(1).mean(1)
            + (fused_y_axis - target_y).abs().flatten(1).mean(1)
        )
    return total / float(len(scales))


def _ssim_map(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    window: int,
) -> torch.Tensor:
    if window < 3 or window % 2 == 0:
        raise ValueError("SSIM window must be an odd integer >= 3")
    coords = torch.arange(
        window, device=prediction.device, dtype=prediction.dtype
    ) - (window - 1) / 2.0
    vector = torch.exp(-0.5 * (coords / 1.5).square())
    vector = vector / vector.sum()
    kernel = (vector[:, None] * vector[None, :]).view(1, 1, window, window)
    padding = window // 2
    mean_p = F.conv2d(prediction, kernel, padding=padding)
    mean_r = F.conv2d(reference, kernel, padding=padding)
    var_p = (
        F.conv2d(prediction.square(), kernel, padding=padding) - mean_p.square()
    ).clamp_min(0.0)
    var_r = (
        F.conv2d(reference.square(), kernel, padding=padding) - mean_r.square()
    ).clamp_min(0.0)
    covariance = (
        F.conv2d(prediction * reference, kernel, padding=padding)
        - mean_p * mean_r
    )
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2.0 * mean_p * mean_r + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_p.square() + mean_r.square() + c1)
        * (var_p + var_r + c2)
    ).clamp_min(1e-12)
    return (numerator / denominator).clamp(-1.0, 1.0)


def _structure_energy(
    fused_y: torch.Tensor,
    visible_y: torch.Tensor,
    infrared_y: torch.Tensor,
    visible_weight: torch.Tensor,
    infrared_weight: torch.Tensor,
    window: int,
) -> torch.Tensor:
    visible_ssim = _ssim_map(fused_y, visible_y, window)
    infrared_ssim = _ssim_map(fused_y, infrared_y, window)
    loss = (
        visible_weight * (1.0 - visible_ssim)
        + infrared_weight * (1.0 - infrared_ssim)
    )
    return loss.flatten(1).mean(1)


def _saliency_energy(
    fused_y: torch.Tensor,
    visible_y: torch.Tensor,
    infrared_y: torch.Tensor,
    visible_weight: torch.Tensor,
    infrared_weight: torch.Tensor,
) -> torch.Tensor:
    loss = (
        visible_weight * (fused_y - visible_y).square()
        + infrared_weight * (fused_y - infrared_y).square()
    )
    return loss.flatten(1).mean(1)


class FusionEnergy(nn.Module):
    """Weighted scalar energy used by both inference and Stage-III training."""

    def __init__(
        self,
        intensity_weight: float = 1.0,
        gradient_weight: float = 1.0,
        structure_weight: float = 1.0,
        saliency_weight: float = 1.0,
        saliency_temperature: float = 0.5,
        ssim_window: int = 11,
    ):
        super().__init__()
        self.intensity_weight = float(intensity_weight)
        self.gradient_weight = float(gradient_weight)
        self.structure_weight = float(structure_weight)
        self.saliency_weight = float(saliency_weight)
        weights = (
            self.intensity_weight,
            self.gradient_weight,
            self.structure_weight,
            self.saliency_weight,
        )
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError("Fusion-energy weights must be non-negative and not all zero")
        self.saliency_temperature = float(saliency_temperature)
        if self.saliency_temperature <= 0.0:
            raise ValueError("saliency_temperature must be positive")
        self.ssim_window = int(ssim_window)
        if self.ssim_window < 3 or self.ssim_window % 2 == 0:
            raise ValueError("ssim_window must be an odd integer >= 3")

    @staticmethod
    def _validate(*images: torch.Tensor) -> None:
        reference = images[0].shape[0], images[0].shape[-2:]
        for image in images:
            if not torch.is_floating_point(image):
                raise TypeError("FusionEnergy inputs must be floating point")
            if not torch.isfinite(image).all():
                raise FloatingPointError("FusionEnergy received non-finite input")
            if (image.shape[0], image.shape[-2:]) != reference:
                raise ValueError("FusionEnergy inputs must share batch/spatial shape")

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        visible_01: torch.Tensor,
        infrared_01: torch.Tensor,
        fused_prediction_01: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate(visible_01, infrared_01, fused_prediction_01)
        visible_y = _luminance(visible_01.float()).clamp(0.0, 1.0)
        infrared_y = _luminance(infrared_01.float()).clamp(0.0, 1.0)
        fused_y = _luminance(fused_prediction_01.float())
        visible_weight, infrared_weight = saliency_weights(
            visible_y, infrared_y, self.saliency_temperature
        )

        intensity = _intensity_energy(fused_y, visible_y, infrared_y)
        gradient = _gradient_energy(fused_y, visible_y, infrared_y)
        structure = _structure_energy(
            fused_y,
            visible_y,
            infrared_y,
            visible_weight,
            infrared_weight,
            self.ssim_window,
        )
        saliency = _saliency_energy(
            fused_y,
            visible_y,
            infrared_y,
            visible_weight,
            infrared_weight,
        )
        energy = (
            self.intensity_weight * intensity
            + self.gradient_weight * gradient
            + self.structure_weight * structure
            + self.saliency_weight * saliency
        )
        if not torch.isfinite(energy).all():
            raise FloatingPointError("Direct fusion energy is non-finite")
        return energy, {
            "intensity_consistency": intensity,
            "gradient_consistency": gradient,
            "structure_consistency": structure,
            "saliency_consistency": saliency,
        }
