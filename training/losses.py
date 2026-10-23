"""Source reconstruction and fusion-task losses for Stage III."""

import torch
import torch.nn.functional as F

from RF.fusion_energy import _luminance

def _ssim_1d(kernel: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Compute SSIM map along one spatial dimension.  x, y: [N, 1, L]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    weight = kernel.view(1, 1, -1).to(x.device, x.dtype)
    pad = kernel.shape[-1] // 2
    mu_x = F.conv1d(x, weight, padding=pad)
    mu_y = F.conv1d(y, weight, padding=pad)
    mu_xx = mu_x.square()
    mu_yy = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_xx = F.conv1d(x.square(), weight, padding=pad) - mu_xx
    sigma_yy = F.conv1d(y.square(), weight, padding=pad) - mu_yy
    sigma_xy = F.conv1d(x * y, weight, padding=pad) - mu_xy
    return torch.clamp(
        (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        / ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)),
        min=0.0,
        max=1.0,
    )


def differentiable_ssim_map(
    x_01: torch.Tensor, y_01: torch.Tensor
) -> torch.Tensor:
    """Per-pixel separable SSIM map in [0,1] (1 = identical), B×1×H×W."""
    if x_01.ndim != 4 or x_01.shape[1] != 1:
        raise ValueError("SSIM expects B×1×H×W luminance inputs")
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss = (-0.5 * (coords / sigma).square()).exp()
    gauss = gauss / gauss.sum()
    batch, channel, height, width = x_01.shape

    # Along width: [B, C, H, W] -> [B*C*H, 1, W]
    x_w = x_01.reshape(batch * channel * height, 1, width)
    y_w = y_01.reshape(batch * channel * height, 1, width)
    ssim_w_map = _ssim_1d(gauss, x_w, y_w)
    ssim_w = ssim_w_map.view(batch, channel, height, width)

    # Along height: [B, C, H, W] -> permute to [B, C, W, H] -> [B*C*W, 1, H]
    x_h = x_01.permute(0, 1, 3, 2).reshape(batch * channel * width, 1, height)
    y_h = y_01.permute(0, 1, 3, 2).reshape(batch * channel * width, 1, height)
    ssim_h_map = _ssim_1d(gauss, x_h, y_h)
    ssim_h = ssim_h_map.view(batch, channel, width, height).permute(0, 1, 3, 2)

    return ssim_w * ssim_h


def differentiable_ssim(x_01: torch.Tensor, y_01: torch.Tensor) -> torch.Tensor:
    """Per-sample separable SSIM in [0,1] on luminance.

    Returns (1 - SSIM) scalar so the minimisation direction is loss-like.
    """
    ssim_map = differentiable_ssim_map(x_01, y_01)
    return 1.0 - ssim_map.flatten(1).mean(1).mean()

def fusion_task_loss(fused_01, visible_01, infrared_01):
    """Existing intensity/SSIM/gradient objectives in the deployed gray domain.

    Preserve the fusion loss weights 4/1/10 and SSIM window 48. Color and
    histogram-mask terms are deliberately excluded from this gray-output trial.
    """
    from util.loss import ssim

    fused, visible, infrared = [
        _luminance(image.float()) for image in (fused_01, visible_01, infrared_01)
    ]
    intensity = F.l1_loss(fused, torch.maximum(visible, infrared))
    structure = (ssim(visible, fused, window_size=48, val_range=1)
                 + ssim(infrared, fused, window_size=48, val_range=1))
    kernels = fused.new_tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    ]).unsqueeze(1)

    def gradient(image):
        return F.conv2d(F.pad(image, (1, 1, 1, 1), mode='replicate'), kernels).abs()

    difference = (gradient(fused) - torch.maximum(gradient(visible), gradient(infrared))).abs()
    edge = difference.mean(dim=(0, 2, 3)).sum()
    return 4 * intensity + structure + 10 * edge, {
        'task_intensity_loss': intensity,
        'task_structure_loss': structure,
        'task_gradient_loss': edge,
    }
