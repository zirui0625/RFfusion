"""RF-aligned guidance for the conference sampler effective image update."""

from typing import Dict, Tuple

import torch

from util.image import normalize_luminance_01


def _l2_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1).norm(p=2, dim=1)


def apply_rf_aligned_guidance(
    effective_update: torch.Tensor,
    energy_gradient: torch.Tensor,
    scale: float,
    eps: float = 1e-8,
    normalize_gradient: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply positive-energy guidance; raw mode is for normalization ablation."""
    if effective_update.shape != energy_gradient.shape:
        raise ValueError("Effective update and energy gradient must share shape")
    if float(scale) < 0.0:
        raise ValueError("guidance scale must be non-negative")
    if normalize_gradient:
        update_norm = _l2_per_sample(effective_update)
        gradient_norm = _l2_per_sample(energy_gradient)
        ratio = (update_norm / (gradient_norm + float(eps))).view(-1, 1, 1, 1)
        correction = float(scale) * ratio * energy_gradient.float()
    else:
        correction = float(scale) * energy_gradient.float()
    guided_update = effective_update.float() - correction
    if not torch.isfinite(guided_update).all():
        raise FloatingPointError("RF-aligned effective update is non-finite")
    return guided_update.to(effective_update.dtype), correction


def effective_rf_aligned_step(
    current_image_m11: torch.Tensor,
    effective_update_m11: torch.Tensor,
    visible_01: torch.Tensor,
    infrared_01: torch.Tensor,
    fusion_energy,
    *,
    dt: float = 1.0,
    guidance_scale: float = 0.0,
    guidance_loss_scale: float = 2048.0,
    normalize_gradient: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Guide the released conference RF+EM image-space update.

    effective_update_m11 is the EM fusion prior before the DDIM-style
    reconstruction. The fusion gradient is evaluated on the candidate fused
    image and detached before the outer Stage-III backward. By default it is
    L2-normalized per sample against this update; ablation can use it raw.
    """
    dt = float(dt)
    guidance_scale = float(guidance_scale)
    guidance_loss_scale = float(guidance_loss_scale)
    if dt != 1.0:
        raise ValueError("effective RF-aligned guidance requires one-step dt=1")
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if guidance_loss_scale <= 0.0:
        raise ValueError("guidance_loss_scale must be positive")
    if current_image_m11.shape != effective_update_m11.shape:
        raise ValueError("Current image and effective update must share shape")

    base_fused_m11 = current_image_m11 + effective_update_m11 * dt
    if guidance_scale == 0.0:
        energy_gradient = torch.zeros_like(
            effective_update_m11, dtype=torch.float32
        )
        correction = torch.zeros_like(
            effective_update_m11, dtype=torch.float32
        )
        guided_update = effective_update_m11
        with torch.no_grad():
            energy_before, term_values = fusion_energy(
                visible_01.float(),
                infrared_01.float(),
                normalize_luminance_01(base_fused_m11.float()),
            )
    else:
        with torch.enable_grad():
            fused_proxy = base_fused_m11.detach().float().requires_grad_(True)
            energy_before, term_values = fusion_energy(
                visible_01.float(),
                infrared_01.float(),
                normalize_luminance_01(fused_proxy),
            )
            energy_gradient = torch.autograd.grad(
                energy_before.sum() * guidance_loss_scale,
                fused_proxy,
                retain_graph=False,
                create_graph=False,
            )[0].detach().float() / guidance_loss_scale
        if not torch.isfinite(energy_gradient).all():
            raise FloatingPointError("Fusion-energy gradient is non-finite")
        if torch.any(_l2_per_sample(energy_gradient) <= 0.0):
            raise FloatingPointError("Fusion-energy gradient vanished")
        guided_update, correction = apply_rf_aligned_guidance(
            effective_update_m11,
            energy_gradient,
            guidance_scale,
            normalize_gradient=normalize_gradient,
        )

    fused_image_m11 = current_image_m11 + guided_update * dt
    fused_native_01 = (fused_image_m11.float() + 1.0) * 0.5
    fused_luminance_01 = normalize_luminance_01(fused_image_m11.float())
    fused_deployed_01 = fused_luminance_01.repeat(1, 3, 1, 1)
    with torch.no_grad():
        energy_after, _ = fusion_energy(
            visible_01.float(),
            infrared_01.float(),
            normalize_luminance_01(fused_image_m11.detach()),
        )
    update_norm = _l2_per_sample(effective_update_m11)
    gradient_norm = _l2_per_sample(energy_gradient)
    correction_norm = _l2_per_sample(correction)
    diagnostics = {
        "effective_update": effective_update_m11.detach(),
        "guided_velocity": guided_update.detach(),
        "fused_image": fused_image_m11,
        "fused_native_01": fused_native_01,
        "fusion_energy_image_01": fused_luminance_01,
        "fused_deployed_01": fused_deployed_01,
        "guidance_energy": energy_before.detach().mean(),
        "guidance_energy_after": energy_after.detach().mean(),
        "guidance_velocity_norm": update_norm.detach().mean(),
        "guidance_gradient_norm": gradient_norm.detach().mean(),
        "guidance_correction_norm": correction_norm.detach().mean(),
        "guidance_relative_correction": (
            correction_norm / (update_norm + 1e-8)
        ).detach().mean(),
    }
    diagnostics.update(
        {
            "guidance_{}".format(name): value.detach().mean()
            for name, value in term_values.items()
        }
    )
    return fused_image_m11, diagnostics
