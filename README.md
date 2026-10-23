# Efficient Rectified Flow for Image Fusion

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)

### Efficient Rectified Flow for Image Fusion [NeurIPS 2025]

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/RFframework.png" width="90%">
</div>
<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/VAEtraining.png" width="90%">
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
Download the Rectified Flow checkpoint from [here](https://github.com/gnobitab/RectifiedFlow), we use 'checkpoint12.pth' for sampling, and put it in './model'. Our pretrained VAE model (stage I+II) can be found in [here](https://drive.google.com/file/d/10Rmz6YtGnM2qHk1QfjCY9eEFkh0gsvVZ/view?usp=drive_link), also put it in './model'. Our pretrained VAE model (after stage III) can be found in [here](https://drive.google.com/file/d/1NYLQW94FjOQ_q7XX6fWVRM-O-j_fcM5E/view?usp=drive_link). You can test our method through
```
python sample.py
```
## Train

### Stage I
You can download the training data from [LLVIP](https://bupt-ai-cz.github.io/LLVIP/), [MSRS](https://github.com/Linfeng-Tang/MSRS), and place it in `./data`. You can download the initial VAE model (f=4, VQ) from [LDM](https://github.com/CompVis/latent-diffusion) and place it in `./model`.

```
CUDA_VISIBLE_DEVICES=0 python train_vae.py -b ./vae_config.yaml -t -r ./model/model.ckpt --gpus 0,
```
### Stage II
You can download the training data from [MSRS](https://github.com/Linfeng-Tang/MSRS), and place it in `./data`.

```
CUDA_VISIBLE_DEVICES=0 python train_fusion.py --rf_config configs/stage2.yaml
```
### Stage III
Set the Stage I–II VAE and paired LLVIP/MSRS directories in `configs/stage3.yaml`. 
```
CUDA_VISIBLE_DEVICES=0 python train_stage3.py --config configs/stage3.yaml
```
### Relevant Datasets
We evaluate image fusion and downstream performance using the following datasets.

| Task | Dataset | Original source |
| --- | --- | --- |
| Infrared–visible fusion and object detection | M$^3$FD | [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) |
| Infrared–visible fusion | TNO | [TNO Image Fusion Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) |
| Infrared–visible fusion and efficiency evaluation | RoadScene | [RoadScene](https://github.com/hanna-xu/RoadScene) |
| Multi-exposure fusion | MEFB | [MEFB](https://github.com/xingchenzhang/MEFB) |
| Multi-focus fusion | Lytro | [Lytro Multi-focus Image Dataset](https://github.com/mnnejati/LytroDataset) |
| Multi-focus fusion | MFFW | [MFFW dataset information](https://github.com/shuangxu96/MFF-SSIM) |
| Multi-focus fusion | MFI-WHU | [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU) |
| Semantic segmentation | FMB | [SegMiF](https://github.com/JinyuanLiu-CV/SegMiF) |

For multi-exposure fusion、multi-focus fusion and medical fusion tasks，you can download our testsets in [here](https://drive.google.com/file/d/1vE262kD5IwTXfEj53fqcWzKSdvApCHN4/view?usp=sharing).

## Results
<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/IVIF.png" width="100%">
</div>

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/MFF+MEF.png" width="100%">
</div>

<div align=center>
<img src="https://github.com/zirui0625/RFfusion/blob/main/figures/MIF.png" width="100%">
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
