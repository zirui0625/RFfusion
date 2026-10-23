"""Stable input VJP for the existing cold-start, one-iteration EM forward.

The deployment output is supplied unchanged. Only the training backward is
recomputed; this is not a replacement fusion operator or an identity STE.
"""

import math

import torch


def stable_em_update(velocity_image, infrared, visible, fft_denominator, rho):
    """Algebraic EM update with C divided out before any large reciprocal.

    C=sqrt(2)/abs(I-p+1e-6), D=min(sqrt(2)/sqrt((p-V)^2+1e-6),2C).
    The original M-step numerator and denominator are divided by C.
    Cold-start F1/F2 are zero; their updated values are not consumed again.
    Inputs retain sampleWithEM's two /255 operations and output *255.
    """
    luminance = (
        .299 * velocity_image[:, 0:1]
        + .587 * velocity_image[:, 1:2]
        + .114 * velocity_image[:, 2:3]
    ) / 255.0 / 255.0
    residual = luminance - visible
    source_difference = infrared - visible
    absolute = (infrared - luminance + 1e-6).abs()
    inverse_c = absolute / math.sqrt(2.0)
    d_over_c = (absolute / (residual.square() + 1e-6).sqrt()).clamp_max(2.0)
    lowpass = torch.fft.ifft2(
        torch.fft.fft2(infrared - luminance) / fft_denominator
    ).real
    numerator = 2 * source_difference + rho * inverse_c * (source_difference - lowpass)
    denominator = 2 + 2 * d_over_c + rho * inverse_c
    return (infrared - numerator / denominator) * 255.0


class _EMInputGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, velocity_image, infrared, visible, fft_denominator, rho, deployed_update):
        ctx.save_for_backward(velocity_image, infrared, visible, fft_denominator)
        ctx.rho = float(rho)
        return deployed_update

    @staticmethod
    def backward(ctx, output_gradient):
        velocity, infrared, visible, denominator = ctx.saved_tensors
        # These small image/FFT calculations do not retain the RF/decoder graph.
        # Float64 avoids the large-reciprocal derivative instability of C.
        with torch.enable_grad(), torch.autocast(device_type=velocity.device.type, enabled=False):
            prediction = velocity.detach().double().requires_grad_(True)
            update = stable_em_update(
                prediction, infrared.double(), visible.double(), denominator.to(torch.complex128), ctx.rho
            ).expand_as(prediction)
            gradient = torch.autograd.grad(update, prediction, output_gradient.double())[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite exact EM input gradient")
        return gradient.to(velocity.dtype), None, None, None, None, None


def attach_exact_em_gradient(velocity_image, infrared, visible, hyperparameters, rho, deployed_update):
    """Attach the true one-step EM input derivative to its original output."""
    denominator = 1 + hyperparameters['fft_k1sq'] + hyperparameters['fft_k2sq']
    return _EMInputGradient.apply(
        velocity_image, infrared, visible, denominator, float(rho), deployed_update.detach()
    )


def normalize_training_luminance(image, eps=1e-6):
    """Same min-max values as deployment, including extrema in outer backward."""
    image = image.float()
    luminance = .299 * image[:, 0:1] + .587 * image[:, 1:2] + .114 * image[:, 2:3]
    minimum = luminance.amin(dim=(1, 2, 3), keepdim=True)
    dynamic_range = (luminance.amax(dim=(1, 2, 3), keepdim=True) - minimum).clamp_min(eps)
    return (luminance - minimum) / dynamic_range
