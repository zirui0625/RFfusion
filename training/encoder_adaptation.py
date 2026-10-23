"""Adapt the VAE encoder through source reconstruction and guided fusion."""

import torch
from torch import nn

from RF.fusion_energy import FusionEnergy, _luminance
from RF.rf_aligned_guidance import effective_rf_aligned_step
from RF.sampleWithEM import compute_alphas_cumprod, p_sample
from guided_diffusion.EM_onestep import EM_Initial
from training.em_backward import attach_exact_em_gradient, normalize_training_luminance
from training.losses import differentiable_ssim, fusion_task_loss


def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class EncoderAdaptation(nn.Module):
    def __init__(self, autoencoder, rf_model, rf_sde, sampling,
                 lambda_reconstruction=1.0, lambda_commitment=0.25,
                 lambda_task=8.07, reconstruction_ssim_weight=0.2,
                 rf_native_size=(64, 64), checkpoint_decoder=False):
        super().__init__()
        self.autoencoder = freeze_module(autoencoder.float())
        self.rf_model = freeze_module(rf_model.float())
        self.rf_sde = rf_sde
        for parameter in self.autoencoder.encoder.parameters():
            parameter.requires_grad_(True)
        self.lambda_reconstruction = float(lambda_reconstruction)
        self.lambda_commitment = float(lambda_commitment)
        self.lambda_task = float(lambda_task)
        self.reconstruction_ssim_weight = float(reconstruction_ssim_weight)
        self.rf_native_size = tuple(rf_native_size)
        self.time_value = 0.001
        self.guidance_scale = float(sampling['guidance_scale'])
        self.guidance_loss_scale = float(sampling.get('guidance_loss_scale', 2048.0))
        self.fusion_energy = FusionEnergy(
            intensity_weight=float(sampling['guidance_intensity_weight']),
            gradient_weight=float(sampling['guidance_gradient_weight']),
            structure_weight=float(sampling['guidance_structure_weight']),
            saliency_weight=float(sampling['guidance_saliency_weight']),
            saliency_temperature=float(sampling.get('select_temperature', 0.5)),
            ssim_window=int(sampling.get('guidance_ssim_window', 11)),
        )
        self.alphas_cumprod, self.alphas_cumprod_prev = compute_alphas_cumprod([0.1] * 1000)
        if int(rf_sde.sample_N) != 1 or float(rf_sde.sigma_t(self.time_value)) != 0:
            raise ValueError('Stage III requires one-step sampling with sigma_variance=0')
        if checkpoint_decoder:
            from torch.utils.checkpoint import checkpoint
            decoder_forward = self.autoencoder.decoder.forward

            def memory_efficient_decode(latent):
                if torch.is_grad_enabled() and latent.requires_grad:
                    return checkpoint(decoder_forward, latent, use_reentrant=False)
                return decoder_forward(latent)

            self.autoencoder.decoder.forward = memory_efficient_decode

    def enforce_frozen_eval(self):
        self.rf_model.eval()
        self.autoencoder.eval()
        self.autoencoder.encoder.train(self.training)

    def encoder_parameters(self):
        return self.autoencoder.encoder.parameters()

    def encode(self, image):
        hidden = self.autoencoder.encoder(image * 2.0 - 1.0)
        prequant = self.autoencoder.quant_conv(hidden)
        quantized, commitment, _ = self.autoencoder.quantize(prequant)
        return quantized * 4.0, commitment

    def decode(self, latent):
        return self.autoencoder.decode(latent / 4.0)

    def reconstruction_loss(self, decoded, source):
        pixel = torch.nn.functional.smooth_l1_loss(decoded.float(), source.float())
        structure = differentiable_ssim(_luminance(decoded.float()), _luminance(source.float()))
        return pixel + self.reconstruction_ssim_weight * structure

    def fused_image(self, latent, infrared, visible):
        if tuple(latent.shape[-2:]) != self.rf_native_size:
            raise ValueError('Training latent must have native size {}'.format(self.rf_native_size))
        batch = latent.shape[0]
        time_rf = torch.full((batch,), self.time_value * 999.0,
                             device=latent.device, dtype=latent.dtype)
        velocity_image = self.decode(self.rf_model(latent, time_rf))
        time = torch.full((batch,), self.time_value, device=latent.device, dtype=latent.dtype)
        infrared_condition = infrared.float() * 2.0 - 1.0
        visible_condition = _luminance(visible.float()) * 2.0 - 1.0
        with torch.no_grad():
            hyperparameters = EM_Initial(infrared_condition)
            prediction, _ = p_sample(
                x=velocity_image.detach(), t=time, bfHP=hyperparameters,
                infrared=infrared_condition, visible=visible_condition,
                lamb=0.5, rho=0.01, alphas_cumprod=self.alphas_cumprod,
                alphas_cumprod_prev=self.alphas_cumprod_prev,
            )
            deployed_update = prediction['pred_xstart'].expand_as(velocity_image)
        effective_update = attach_exact_em_gradient(
            velocity_image, infrared_condition, visible_condition,
            hyperparameters, 0.01, deployed_update,
        )
        _, diagnostics = effective_rf_aligned_step(
            self.decode(latent), effective_update,
            (visible_condition + 1.0) * 0.5, (infrared_condition + 1.0) * 0.5,
            self.fusion_energy, dt=1.0, guidance_scale=self.guidance_scale,
            guidance_loss_scale=self.guidance_loss_scale,
        )
        fused = normalize_training_luminance(diagnostics['fused_image']).repeat(1, 3, 1, 1)
        return fused, diagnostics

    def forward(self, visible, infrared):
        self.enforce_frozen_eval()
        visible, infrared = visible.float(), infrared.float()
        infrared_rgb = infrared.repeat(1, 3, 1, 1) if infrared.shape[1] == 1 else infrared
        latents, reconstruction, commitments = [], [], []
        for source in (visible, infrared_rgb):
            latent, commitment = self.encode(source)
            latents.append(latent)
            reconstruction.append(self.reconstruction_loss((self.decode(latent).float() + 1.0) * 0.5, source))
            commitments.append(commitment.float())
        fused, diagnostics = self.fused_image(latents[0], infrared, visible)
        task, terms = fusion_task_loss(fused, visible, infrared)
        rec = torch.stack(reconstruction).mean()
        commit = torch.stack(commitments).mean()
        total = self.lambda_reconstruction * rec + self.lambda_commitment * commit + self.lambda_task * task
        return {
            'loss': total, 'reconstruction_loss': rec, 'commitment': commit,
            'fusion_training_loss': task, 'visible_reconstruction': reconstruction[0],
            'infrared_reconstruction': reconstruction[1], 'fused_image': fused,
            **terms,
            **{key: value for key, value in diagnostics.items()
               if key.startswith('guidance_') and torch.is_tensor(value) and value.numel() == 1},
        }
