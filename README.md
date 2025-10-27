# Efficient Rectified Flow for Image Fusion

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.11-%237732a8)](https://pytorch.org/)

### Efficient Rectified Flow for Image Fusion [NeurIPS 2025]

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/pipeline.png" width="90%">
</div>

## Updates
[2025-9-25] Our paper has been accepted by NeurIPS 2025. You can find our paper [here](https://arxiv.org/pdf/2509.16549).  

## Environment
```
# create virtual environment
conda create -n RFfusion python=3.8
conda activate RFfusion
# install requirements
pip install -r requirements.txt
```
## Test
Download the Rectified Flow checkpoint from [here](https://github.com/gnobitab/RectifiedFlow), we use 'checkpoint12.pth' for sampling, and put it in './model'. Our pretrained VAE model can be found in [here](https://drive.google.com/file/d/10Rmz6YtGnM2qHk1QfjCY9eEFkh0gsvVZ/view?usp=drive_link), also put it in './model'. You can test our method through
```
python sample.py
```
## Train

# Stage I
```
CUDA_VISIBLE_DEVICES=0 python train_vae.py -b ./vae_config.yaml -t -r ./model/model.ckpt --gpus 0,
```
## Results

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/result1.png" width="100%">
</div>

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/result2.png" width="100%">
</div>


## Citation
```
@article{wang2025efficient,
  title={Efficient Rectified Flow for Image Fusion},
  author={Wang, Zirui and Zhang, Jiayi and Guan, Tianwei and Zhou, Yuhan and Li, Xingyuan and Dong, Minjing and Liu, Jinyuan},
  journal={arXiv preprint arXiv:2509.16549},
  year={2025}
}
```
## Contact
If you have any questions, feel free to contact me through <code style="background-color: #f0f0f0;">ziruiwang0625@gmail.com</code>。
## Acknowledgement
Our codes are based on [Rectified Flow](https://github.com/gnobitab/RectifiedFlow), [LDM](https://github.com/CompVis/latent-diffusion), [DDFM](https://github.com/Zhaozixiang1228/MMIF-DDFM), thanks for their contribution.
