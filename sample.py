from functools import partial
import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from RF.models.utils import create_model
from RF import sde_lib
from RF.sampling import get_sampling_fn
from RF.models import ncsnpp
from RF.models.ema import ExponentialMovingAverage
from RF.utils import save_checkpoint, restore_checkpoint

from util.logger import get_logger
from util.common import instantiate_from_config, load_model, decode_first_stage, encode_first_stage
import cv2
import numpy as np
from skimage.io import imsave
import warnings

import time

warnings.filterwarnings('ignore')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_config', type=str, default='./rf_config.yaml')
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Load configurations
    rf_config = load_yaml(args.rf_config)

    # Device setting
    device = rf_config['device']
    logger.info(f"Device set to {device}.")

    # Load model
    model = create_model(rf_config)
    ema = ExponentialMovingAverage(model.parameters(), decay=rf_config['model']['ema_rate'])
    state = dict(model=model, ema=ema, step=0)
    ckpt_path = rf_config['model']['path']
    state = restore_checkpoint(ckpt_path, state, device = device)
    ema.copy_to(model.parameters())
    model = model.to(device)
    model.eval()

    # load vqvae
    if rf_config['autoencoder'] is not None:
        ae_ckpt_path = rf_config['autoencoder']['ckpt_path']
        assert ae_ckpt_path is not None
        print(f'Loading AutoEncoder model from {ae_ckpt_path}...')
        autoencoder = instantiate_from_config(rf_config['autoencoder']).cuda()
        load_model(autoencoder, ae_ckpt_path, device)
        autoencoder.eval()
        if rf_config['autoencoder']['use_fp16']:
            autoencoder = autoencoder.half()
        else:
            autoencoder = autoencoder

    # Load rectflow sampler
    sde = sde_lib.RectifiedFlow(init_type=rf_config["sampling"]["init_type"], noise_scale=rf_config["sampling"]["init_noise_scale"],
                                use_ode_sampler=rf_config["sampling"]["use_ode_sampler"],
                                sigma_var=rf_config["sampling"]["sigma_variance"], ode_tol=rf_config["sampling"]["ode_tol"],
                                sample_N=rf_config["sampling"]["sample_N"])
    sampling_eps = 1e-3

    shape = (rf_config['eval']['batch_size'],
            rf_config['data']['num_channels'],
            rf_config['data']['image_size'],
            rf_config['data']['image_size'])  
    if rf_config["data"]["centered"]:
        inverse_scaler =  lambda x: (x + 1.) / 2.
    else:
        inverse_scaler =  lambda x: x  
    sample_fn = get_sampling_fn(rf_config, sde, shape, inverse_scaler, sampling_eps)

    # Working directory
    test_folder = rf_config['eval']['path']
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    
    total_time = 0.0
    i = 0
    for img_name in os.listdir(os.path.join(test_folder, "ir")):
        inf_img = image_read(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                      np.newaxis, np.newaxis, ...] / 255.0
        vis_img = image_read(os.path.join(test_folder, "vi", img_name), mode='GRAY')[
                      np.newaxis, np.newaxis, ...] / 255.0
        x_start = image_read(os.path.join(test_folder, "vi", img_name), mode='RGB')[
                      np.newaxis, np.newaxis, ...] / 255.0

        inf_img = inf_img * 2 - 1
        vis_img = vis_img * 2 - 1 
        x_start = x_start * 2 - 1

        inf_img = ((torch.FloatTensor(inf_img))[:, :, :, :]).to(device)
        vis_img = ((torch.FloatTensor(vis_img))[:, :, :, :]).to(device)
        x_start = ((torch.FloatTensor(x_start))[:, :, :, :]).to(device)
        x_start = x_start.squeeze(dim=1).permute(0, 3, 1, 2)

        original_size = inf_img.shape[2:4]
        inf_img = F.interpolate(inf_img, rf_config['sampling']['size'], mode='bicubic', align_corners=False)
        vis_img = F.interpolate(vis_img, rf_config['sampling']['size'], mode='bicubic', align_corners=False)
        x_start = F.interpolate(x_start, rf_config['sampling']['size'], mode='bicubic', align_corners=False)

        x_start = encode_first_stage(x_start, autoencoder)

        assert inf_img.shape == vis_img.shape
        logger.info(f"Inference for image {i}")

        # Sampling
        seed = 3407
        torch.manual_seed(seed)

        if i > 0:
            start_time = time.time()

        with torch.no_grad():
            sample, _ = sample_fn(model, x_start, inf_img, vis_img, autoencoder)

        if i > 0:
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            logger.info(f"Inference time for image {i}: {elapsed:.4f} seconds")

        sample = F.interpolate(sample, (original_size[0], original_size[1]), mode='bicubic', align_corners=False)

        sample = sample.detach().cpu().squeeze().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = np.clip(sample, 0, 1) 
        sample = ((sample) * 255).astype(np.uint8)
        imsave(os.path.join(out_path, "{}.png".format(img_name.split(".")[0])), sample)
        i = i + 1

if i > 1:
    avg_time = total_time / (i - 1)
    logger.info(f"Average inference time per image (excluding first): {avg_time:.4f} seconds")














