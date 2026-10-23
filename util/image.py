"""Luminance conversion for deployed fusion outputs."""

import torch

def _gray(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError("Expected BCHW image with one or three channels")
    if image.shape[1] == 1:
        return image
    return (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )


def normalize_luminance_01(
    image: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Map native sampler luminance to [0, 1] with frozen range statistics.

    This is the differentiable counterpart of sample.py's final per-image
    min-max normalization. Gradients flow through luminance, while the minimum
    and dynamic range are first-order constants.
    """
    luminance = _gray(image.float())
    minimum = luminance.amin(dim=(1, 2, 3), keepdim=True).detach()
    dynamic_range = (
        luminance.amax(dim=(1, 2, 3), keepdim=True).detach() - minimum
    ).clamp_min(float(eps))
    normalized = (luminance - minimum) / dynamic_range
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("Non-finite normalized sampler luminance")
    return normalized
