# coding=utf-8
"""One-step image fusion with RF-aligned guidance."""

import torch

import RF.models.utils as mutils
from RF.fusion_energy import FusionEnergy
from RF.rf_aligned_guidance import effective_rf_aligned_step
from RF.sampleWithEM import compute_alphas_cumprod, p_sample
from guided_diffusion.EM_onestep import EM_Initial
from util.common import decode_first_stage


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    del eps
    if config["sampling"]["method"].lower() != "rectified_flow":
        raise ValueError("RF-aligned guidance requires rectified_flow")
    sampling = config["sampling"]
    return get_rectified_flow_sampler(
        sde=sde,
        shape=shape,
        inverse_scaler=inverse_scaler,
        device=config["device"],
        guidance_scale=sampling.get("guidance_scale", 0.01),
        guidance_loss_scale=sampling.get("guidance_loss_scale", 2048.0),
        intensity_weight=sampling.get("guidance_intensity_weight", 0.15),
        gradient_weight=sampling.get("guidance_gradient_weight", 0.02),
        structure_weight=sampling.get("guidance_structure_weight", 0.03),
        saliency_weight=sampling.get("guidance_saliency_weight", 0.8),
        saliency_temperature=sampling.get("select_temperature", 0.5),
        ssim_window=int(sampling.get("guidance_ssim_window", 11)),
        normalize_gradient=sampling.get("guidance_normalize_gradient", True),
    )


def get_rectified_flow_sampler(
    sde,
    shape,
    inverse_scaler,
    device="cuda:0",
    lamb=0.5,
    rho=0.01,
    guidance_scale=0.01,
    guidance_loss_scale=2048.0,
    intensity_weight=0.15,
    gradient_weight=0.02,
    structure_weight=0.03,
    saliency_weight=0.8,
    saliency_temperature=0.5,
    ssim_window=11,
    normalize_gradient=True,
):
    if sde.use_ode_sampler != "euler":
        raise ValueError("RF-aligned guidance requires Euler sampling")
    if int(sde.sample_N) != 1:
        raise ValueError("RF-aligned guidance requires one-step sampling")
    if float(sde.sigma_t(1e-3)) != 0.0:
        raise ValueError("RF-aligned guidance requires sigma_variance=0")
    fusion_energy = FusionEnergy(
        intensity_weight=float(intensity_weight),
        gradient_weight=float(gradient_weight),
        structure_weight=float(structure_weight),
        saliency_weight=float(saliency_weight),
        saliency_temperature=float(saliency_temperature),
        ssim_window=int(ssim_window),
    ).to(device)
    alphas_cumprod, alphas_cumprod_prev = compute_alphas_cumprod(
        [0.1] * 1000
    )

    def euler_sampler(
        model,
        z=None,
        infrared=None,
        visible=None,
        autoencoder=None,
    ):
        if z is None:
            z0 = sde.get_z0(
                torch.zeros(shape, device=device), train=False
            ).to(device)
            x = z0.detach().clone()
        else:
            x = z
        if infrared is None or visible is None or autoencoder is None:
            raise ValueError("RF-aligned guidance requires both sources and VAE")

        model_fn = mutils.get_model_fn(model, train=False)
        dt = 1.0 / float(sde.sample_N)
        num_t = 1e-3
        t = torch.full(
            (x.shape[0],), num_t, device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            velocity_image = decode_first_stage(
                model_fn(x, t * 999.0), autoencoder
            )
            bfHP = EM_Initial(infrared)
            prediction, _ = p_sample(
                x=velocity_image,
                t=t,
                bfHP=bfHP,
                infrared=infrared,
                visible=visible,
                lamb=lamb,
                rho=rho,
                alphas_cumprod=alphas_cumprod,
                alphas_cumprod_prev=alphas_cumprod_prev,
            )
            current_image = decode_first_stage(x, autoencoder)
            # pred_xstart is the EM prior before the DDIM-style reconstruction.
            effective_update = prediction["pred_xstart"].expand_as(
                current_image
            )

        fused_image, _ = effective_rf_aligned_step(
            current_image,
            effective_update,
            ((visible + 1.0) * 0.5).float(),
            ((infrared + 1.0) * 0.5).float(),
            fusion_energy,
            dt=dt,
            guidance_scale=float(guidance_scale),
            guidance_loss_scale=float(guidance_loss_scale),
            normalize_gradient=normalize_gradient,
        )
        return inverse_scaler(fused_image.detach()), 1

    print("Type of Sampler:", sde.use_ode_sampler)
    print("RF-aligned guidance: EM effective-update RF alignment")
    return euler_sampler
