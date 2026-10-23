"""Frozen fusion target path for Stage-II decoder training."""

import numpy as np
import torch
import RF.models.utils as mutils
from RF.sampleWithEM import p_sample, compute_alphas_cumprod
from guided_diffusion.EM_onestep import EM_Initial
from util.common import decode_first_stage


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    return get_rectified_flow_sampler(sde, shape, inverse_scaler, config["device"])


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device = 'cuda:0', lamb=0.5, rho=0.01):
    """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def euler_sampler(model, z=None, infrared=None, visible=None, autoencoder=None):
        """The probability flow ODE sampler with simple Euler discretization.
    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                x = z0.detach().clone()
            else:
                x = z

            model_fn = mutils.get_model_fn(model, train=False)

            dt = 1. / sde.sample_N
            eps = 1e-3 

            for i in range(sde.sample_N):
                num_t = i / sde.sample_N * (sde.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                pred = model_fn(x, t * 999) 
                pred = decode_first_stage(pred, autoencoder)
                
                sigma_t = sde.sigma_t(num_t)
                betas = [0.1] * 1000  
                alphas_cumprod, alphas_cumprod_prev = compute_alphas_cumprod(betas)
                bfHP = EM_Initial(infrared)
                pred_dict, bfHP = p_sample(x=pred, t=t, bfHP=bfHP, infrared=infrared, visible=visible, lamb=lamb,
                            rho=rho,alphas_cumprod=alphas_cumprod, alphas_cumprod_prev=alphas_cumprod_prev)
                pred = pred_dict["sample"]  
              
                x = decode_first_stage(x, autoencoder)  
                num_t = float(num_t) 
                pred_sigma = pred + (sigma_t ** 2) / (2 * (sde.noise_scale ** 2) * ((1. - num_t) ** 2)) * (
                        0.5 * num_t * (1. - num_t) * pred - 0.5 * (2. - num_t) * x)
                x = x + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(
                    device)

            x = inverse_scaler(x)
            nfe = sde.sample_N
            return x, nfe

    return euler_sampler
