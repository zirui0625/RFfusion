import argparse
import copy
import datetime
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from RF.models.utils import create_model
from RF.models.ema import ExponentialMovingAverage
from RF.utils import restore_checkpoint
from RF import sde_lib
from RF.sampling import get_sampling_fn

from util import transforms as T
from util.dataset import SimpleDataSet
from util.loss import fusion_loss
from util.common import (
    create_lr_scheduler,
    decode_first_stage,
    encode_first_stage,
    instantiate_from_config,
    load_model,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VAE decoder with a frozen Rectified Flow fusion sampler."
    )
    parser.add_argument("--data-root", type=str, default="./data/MSRS")
    parser.add_argument("--rf-config", type=str, default="./rf_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-root", type=str, default="./logs")
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-val-images", action="store_true")
    parser.add_argument(
        "--force-ae-fp32",
        action="store_true",
        help="Force the trainable autoencoder to float32 for stable decoder training.",
    )
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_experiment_dir(log_root: str) -> Dict[str, Path]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(log_root) / f"train_fusion_{timestamp}"
    paths = {
        "root": exp_dir,
        "images": exp_dir / "images",
        "weights": exp_dir / "weights",
        "tb": exp_dir / "tensorboard",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train_fusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def rgb_to_grayscale(batch_tensor: torch.Tensor) -> torch.Tensor:
    r = batch_tensor[:, 0:1, :, :]
    g = batch_tensor[:, 1:2, :, :]
    b = batch_tensor[:, 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b

def normalize_for_rf(x: torch.Tensor, centered: bool) -> torch.Tensor:
    return x * 2.0 - 1.0 if centered else x

def denormalize_from_rf(x: torch.Tensor, centered: bool) -> torch.Tensor:
    return (x + 1.0) / 2.0 if centered else x

def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SimpleDataSet(
        args.data_root,
        phase="train",
        transform=T.Compose(
            [
                T.RandomCrop(96),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.ToTensor(),
            ]
        ),
    )
    val_dataset = SimpleDataSet(
        args.data_root,
        phase="eval",
        transform=T.Compose([T.Resize_16(), T.ToTensor()]),
    )

    num_workers = min(os.cpu_count() or 0, args.batch_size if args.batch_size > 1 else 0, args.num_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )
    return train_loader, val_loader

def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

def enable_decoder_only(autoencoder: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for param in autoencoder.parameters():
        param.requires_grad = False

    trainable = []

    if hasattr(autoencoder, "post_quant_conv"):
        autoencoder.post_quant_conv.train()
        for param in autoencoder.post_quant_conv.parameters():
            param.requires_grad = True
            trainable.append(param)

    if hasattr(autoencoder, "decoder"):
        autoencoder.decoder.train()
        for param in autoencoder.decoder.parameters():
            param.requires_grad = True
            trainable.append(param)

    if not trainable:
        for name, param in autoencoder.named_parameters():
            if "decoder" in name or "post_quant_conv" in name:
                param.requires_grad = True
                trainable.append(param)

    if not trainable:
        raise RuntimeError(
            "Could not find decoder parameters. Please adjust enable_decoder_only() to your autoencoder implementation."
        )

    return trainable

def load_autoencoder_pair(rf_config: Dict, device: torch.device, force_fp32: bool = False):
    ae_cfg = rf_config.get("autoencoder")
    if ae_cfg is None:
        raise ValueError("rf_config['autoencoder'] must be provided.")

    ckpt_path = ae_cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("autoencoder.ckpt_path is required.")

    ae_sampler = instantiate_from_config(ae_cfg).to(device)
    load_model(ae_sampler, ckpt_path, str(device))
    freeze_module(ae_sampler)

    ae_train = instantiate_from_config(ae_cfg).to(device)
    load_model(ae_train, ckpt_path, str(device))

    if ae_cfg.get("use_fp16", False) and not force_fp32:
        ae_sampler = ae_sampler.half()
        ae_train = ae_train.half()
    else:
        ae_sampler = ae_sampler.float()
        ae_train = ae_train.float()

    trainable_params = list(enable_decoder_only(ae_train))
    return ae_sampler, ae_train, trainable_params

def load_rf_model(rf_config: Dict, device: torch.device) -> torch.nn.Module:
    rf_config = copy.deepcopy(rf_config)
    rf_config["device"] = str(device)

    model = create_model(rf_config)
    ema = ExponentialMovingAverage(model.parameters(), decay=rf_config["model"]["ema_rate"])
    state = {"model": model, "ema": ema, "step": 0}
    state = restore_checkpoint(rf_config["model"]["path"], state, device=str(device))
    ema.copy_to(model.parameters())
    model.to(device)
    freeze_module(model)
    return model

def build_sde(rf_config: Dict):
    return sde_lib.RectifiedFlow(
        init_type=rf_config["sampling"]["init_type"],
        noise_scale=rf_config["sampling"]["init_noise_scale"],
        use_ode_sampler=rf_config["sampling"]["use_ode_sampler"],
        sigma_var=rf_config["sampling"]["sigma_variance"],
        ode_tol=rf_config["sampling"]["ode_tol"],
        sample_N=rf_config["sampling"]["sample_N"],
    )

def save_checkpoint(path: Path, autoencoder: torch.nn.Module, optimizer, scheduler, epoch: int, args) -> None:
    checkpoint = {
        "model": autoencoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(checkpoint, path)

class FrozenRFSampler:
    def __init__(self, rf_config: Dict, rf_model: torch.nn.Module, ae_sampler: torch.nn.Module, device: torch.device):
        self.rf_config = copy.deepcopy(rf_config)
        self.rf_config["device"] = str(device)
        self.rf_model = rf_model
        self.ae_sampler = ae_sampler
        self.device = device
        self.sde = build_sde(self.rf_config)
        self.sampling_eps = 1e-3
        self.sample_fn_cache = {}

    def _get_sample_fn(self, batch_size: int):
        if batch_size not in self.sample_fn_cache:
            shape = (
                batch_size,
                self.rf_config["data"]["num_channels"],
                self.rf_config["data"]["image_size"],
                self.rf_config["data"]["image_size"],
            )
            centered = self.rf_config["data"].get("centered", False)
            inverse_scaler = (lambda x: (x + 1.0) / 2.0) if centered else (lambda x: x)
            self.sample_fn_cache[batch_size] = get_sampling_fn(
                self.rf_config,
                self.sde,
                shape,
                inverse_scaler,
                self.sampling_eps,
            )
        return self.sample_fn_cache[batch_size]

    @torch.no_grad()
    def sample(self, visible_rgb: torch.Tensor, infrared_rgb: torch.Tensor) -> torch.Tensor:
        batch_size = visible_rgb.shape[0]
        centered = self.rf_config["data"].get("centered", False)
        target_size = tuple(self.rf_config["sampling"]["size"])

        vis_gray = rgb_to_grayscale(visible_rgb)
        ir_gray = rgb_to_grayscale(infrared_rgb)

        x_start = normalize_for_rf(visible_rgb, centered)
        vis_cond = normalize_for_rf(vis_gray, centered)
        ir_cond = normalize_for_rf(ir_gray, centered)

        x_start = F.interpolate(x_start, target_size, mode="bicubic", align_corners=False)
        vis_cond = F.interpolate(vis_cond, target_size, mode="bicubic", align_corners=False)
        ir_cond = F.interpolate(ir_cond, target_size, mode="bicubic", align_corners=False)

        z_start = encode_first_stage(x_start, self.ae_sampler)
        sample_fn = self._get_sample_fn(batch_size)
        fused, _ = sample_fn(self.rf_model, z_start, ir_cond, vis_cond, self.ae_sampler)
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused

def decode_with_trainable_decoder(
    fused_image: torch.Tensor,
    ae_train: torch.nn.Module,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    latent = encode_first_stage(fused_image, ae_train)
    decoded = decode_first_stage(latent, ae_train)
    decoded = torch.clamp(decoded.float(), 0.0, 1.0)
    if decoded.shape[-2:] != output_size:
        decoded = F.interpolate(decoded, output_size, mode="bicubic", align_corners=False)
    return decoded

def run_epoch(
    epoch: int,
    mode: str,
    data_loader: DataLoader,
    rf_sampler: FrozenRFSampler,
    ae_train: torch.nn.Module,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    logger: logging.Logger,
    writer: SummaryWriter,
    save_image_dir: Path = None,
) -> Dict[str, float]:
    is_train = mode == "train"
    ae_train.train(is_train)

    if not is_train:
        ae_train.eval()

    meters = {
        "total": 0.0,
        "ssim": 0.0,
        "max": 0.0,
        "color": 0.0,
        "text": 0.0,
        "mask": 0.0,
    }

    progress = tqdm(data_loader, desc=f"[{mode} epoch {epoch}]", leave=False)
    for step, (image_vis, image_ir, names) in enumerate(progress, start=1):
        image_vis = image_vis.to(device, non_blocking=True).float()
        image_ir = image_ir.to(device, non_blocking=True).float()
        original_size = tuple(image_vis.shape[-2:])

        with torch.no_grad():
            fused_image = rf_sampler.sample(image_vis, image_ir)
            if fused_image.shape[-2:] != original_size:
                fused_image = F.interpolate(fused_image, original_size, mode="bicubic", align_corners=False)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            output = decode_with_trainable_decoder(fused_image, ae_train, original_size)
            loss, loss_ssim, loss_max, loss_color, loss_text, loss_mask = loss_fn(image_vis, image_ir, output)
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                output = decode_with_trainable_decoder(fused_image, ae_train, original_size)
                loss, loss_ssim, loss_max, loss_color, loss_text, loss_mask = loss_fn(image_vis, image_ir, output)

        meters["total"] += float(loss.detach())
        meters["ssim"] += float(loss_ssim.detach())
        meters["max"] += float(loss_max.detach())
        meters["color"] += float(loss_color.detach())
        meters["text"] += float(loss_text.detach())
        meters["mask"] += float(loss_mask.detach())

        averages = {k: v / step for k, v in meters.items()}
        current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
        progress.set_postfix(
            loss=f"{averages['total']:.4f}",
            ssim=f"{averages['ssim']:.4f}",
            max=f"{averages['max']:.4f}",
            color=f"{averages['color']:.4f}",
            text=f"{averages['text']:.4f}",
            mask=f"{averages['mask']:.4f}",
            lr=f"{current_lr:.6f}",
        )

        if save_image_dir is not None and step == 1:
            save_image(output.detach().cpu(), save_image_dir / f"epoch_{epoch:04d}_{names[0]}_output.png")
            save_image(fused_image.detach().cpu(), save_image_dir / f"epoch_{epoch:04d}_{names[0]}_rf_sample.png")

    num_steps = max(len(data_loader), 1)
    stats = {k: v / num_steps for k, v in meters.items()}

    prefix = "train" if is_train else "val"
    for key, value in stats.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)
    if optimizer is not None:
        writer.add_scalar(f"{prefix}/lr", optimizer.param_groups[0]["lr"], epoch)

    logger.info(
        "%s epoch %d | total=%.4f ssim=%.4f max=%.4f color=%.4f text=%.4f mask=%.4f",
        mode,
        epoch,
        stats["total"],
        stats["ssim"],
        stats["max"],
        stats["color"],
        stats["text"],
        stats["mask"],
    )
    return stats

def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    exp_paths = setup_experiment_dir(args.log_root)
    logger = setup_logger(exp_paths["root"] / "train.log")
    writer = SummaryWriter(log_dir=str(exp_paths["tb"]))

    logger.info("Using device: %s", device)
    logger.info("Experiment directory: %s", exp_paths["root"])

    rf_config = load_yaml(args.rf_config)
    train_loader, val_loader = build_dataloaders(args)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    rf_model = load_rf_model(rf_config, device)
    ae_sampler, ae_train, decoder_params = load_autoencoder_pair(
        rf_config, device, force_fp32=args.force_ae_fp32
    )
    rf_sampler = FrozenRFSampler(rf_config, rf_model, ae_sampler, device)

    optimizer = torch.optim.Adam(decoder_params, lr=args.lr)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    loss_fn = fusion_loss()

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        ae_train.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from checkpoint: %s (epoch %d)", args.resume, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        train_stats = run_epoch(
            epoch=epoch,
            mode="train",
            data_loader=train_loader,
            rf_sampler=rf_sampler,
            ae_train=ae_train,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            writer=writer,
        )

        if (epoch + 1) % args.val_every == 0:
            save_dir = exp_paths["images"] if args.save_val_images else None
            val_stats = run_epoch(
                epoch=epoch,
                mode="val",
                data_loader=val_loader,
                rf_sampler=rf_sampler,
                ae_train=ae_train,
                optimizer=optimizer,
                scheduler=None,
                loss_fn=loss_fn,
                device=device,
                logger=logger,
                writer=writer,
                save_image_dir=save_dir,
            )

            if val_stats["total"] < best_val_loss:
                best_val_loss = val_stats["total"]
                save_checkpoint(exp_paths["weights"] / "checkpoint_best.pth", ae_train, optimizer, scheduler, epoch, args)
                logger.info("Saved new best checkpoint with val loss %.4f", best_val_loss)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(exp_paths["weights"] / "checkpoint_last.pth", ae_train, optimizer, scheduler, epoch, args)

    writer.close()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
