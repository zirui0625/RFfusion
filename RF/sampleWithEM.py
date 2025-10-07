import cv2
import numpy as np
import torch
from imageio.v2 import imsave
from skimage.color import rgb2ycbcr
from tqdm import tqdm
import os

from guided_diffusion.EM_onestep import EM_onestep
from util.pytorch_colors import rgb_to_ycbcr, ycbcr_to_rgb, _generic_transform_sk_4d, hed_to_rgb

def extract_and_expand(array, time, target):
    """
    Extract values from the `array` at specific `time` indices and expand them to match the dimensions of `target`.

    Args:
        array: The precomputed array from which values will be extracted (e.g., alphas).
        time: The current timestep.
        target: The target tensor that we want to match the shape of.

    Returns:
        An expanded array of the same shape as `target`.
    """
    array = torch.from_numpy(array).to(target.device)[time.long()].float()
    while array.ndimension() < target.ndimension():
        array = array.unsqueeze(-1)  # Add dimensions if necessary
    return array.expand_as(target)  # Expand to match the target shape

def predict_eps_from_x_start(x_t, t, pred_xstart, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod):
    """
    Compute the predicted epsilon from the given starting image `x_t` and the predicted starting image `pred_xstart`.

    Args:
        x_t: The noisy image at time t.
        t: The current timestep.
        pred_xstart: The predicted clean image (x_start).
        sqrt_recip_alphas_cumprod: The precomputed array for sqrt(1/alpha_t).
        sqrt_recipm1_alphas_cumprod: The precomputed array for sqrt(1/alpha_t - 1).

    Returns:
        The predicted epsilon (eps).
    """
    coef1 = extract_and_expand(sqrt_recip_alphas_cumprod, t, x_t)
    coef2 = extract_and_expand(sqrt_recipm1_alphas_cumprod, t, x_t)
    return (coef1 * x_t - pred_xstart) / coef2


def p_sample(x, t, bfHP, infrared, visible, lamb, rho, alphas_cumprod, alphas_cumprod_prev, eta=0.0):
    """
    A single step of the p_sample function for rectified flow with image fusion and denoising.
    """
    # x is already the model's predicted result (pred)
    pred = x  # This is the prediction passed to the function (no need to call model again)

    # Convert to YCbCr (assuming pred is RGB)
    def rgb_to_ycbcr(input_):
        if input_.dim() == 3:
            input_ = input_.unsqueeze(0)  
        return _generic_transform_sk_4d(rgb2ycbcr)(input_)
    
    def rgb_to_ycbcr_torch(image):
    # Assumes image is (B, C, H, W) and float32 in [0,1] or [-1,1]
    # RGB to YCbCr conversion matrix in PyTorch
        matrix = torch.tensor([
            [ 0.299,  0.587,  0.114],
            [-0.168736, -0.331264,  0.5],
            [ 0.5, -0.418688, -0.081312]
        ], device=image.device, dtype=image.dtype)
    
        bias = torch.tensor([0., 128., 128.], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

        # (B, C, H, W) => (B, H, W, C)
        image = image.permute(0, 2, 3, 1)
        ycbcr = torch.tensordot(image, matrix.T, dims=1) + bias.permute(0, 2, 3, 1)
        # Back to (B, C, H, W)
        ycbcr = ycbcr.permute(0, 3, 1, 2)
        return ycbcr / 255.0


    x_0_hat_ycbcr = rgb_to_ycbcr_torch(pred)/ 255.0  # (-1,1)
    x_0_hat_y = torch.unsqueeze(x_0_hat_ycbcr[:, 0, :, :], 1)

    assert x_0_hat_y.shape[1] == 1
    # Perform fusion step (EM-based or other)
    x_0_hat_y_BF, bfHP = EM_onestep(f_pre=x_0_hat_y,
                                    I=infrared,
                                    V=visible,
                                    HyperP=bfHP,
                                    lamb=lamb,
                                    rho=rho)
    x_0_hat_ycbcr = x_0_hat_y_BF
    # pred = ycbcr_to_rgb(x_0_hat_ycbcr * 255)
    pred = x_0_hat_ycbcr * 255
    # Predict epsilon
    eps = predict_eps_from_x_start(x, t, pred, alphas_cumprod, alphas_cumprod_prev)

    # Calculate alpha_bar and sigma for noise
    alpha_bar = extract_and_expand(alphas_cumprod, t, x)
    alpha_bar_prev = extract_and_expand(alphas_cumprod_prev, t, x)
    sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

    noise = torch.randn_like(x)
    mean_pred = pred * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

    # Add noise if not the final timestep
    sample = mean_pred
    if t != 0:
        sample += sigma * noise

    return {"sample": sample, "pred_xstart": pred}, bfHP

def compute_alphas_cumprod(betas):
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    return alphas_cumprod, alphas_cumprod_prev