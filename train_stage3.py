"""Train the VAE encoder and save a complete VAE for inference."""

import argparse
import copy
import datetime
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from RF import sde_lib
from training.encoder_adaptation import EncoderAdaptation
from training.paired_dataset import PairedIRVIDataset
from util.common import instantiate_from_config, load_model
from util.image import _gray


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_model(config, device):
    from RF.models import ncsnpp  # registers the RF model
    from RF.models.ema import ExponentialMovingAverage
    from RF.models.utils import create_model
    from RF.utils import restore_checkpoint

    with open(config['paths']['rf_config'], encoding='utf-8') as handle:
        rf_config = yaml.safe_load(handle)
    rf_config['device'] = str(device)
    autoencoder = instantiate_from_config(rf_config['autoencoder']).to(device).float()
    load_model(autoencoder, config['paths']['vae_checkpoint'], str(device))
    rf_model = create_model(rf_config)
    ema = ExponentialMovingAverage(rf_model.parameters(), rf_config['model']['ema_rate'])
    state = restore_checkpoint(rf_config['model']['path'],
                               dict(model=rf_model, ema=ema, step=0), device=str(device))
    state['ema'].copy_to(rf_model.parameters())
    sampling = rf_config['sampling']
    sde = sde_lib.RectifiedFlow(
        init_type=sampling['init_type'], noise_scale=sampling['init_noise_scale'],
        use_ode_sampler=sampling['use_ode_sampler'], sigma_var=sampling['sigma_variance'],
        ode_tol=sampling['ode_tol'], sample_N=sampling['sample_N'],
    )
    model = EncoderAdaptation(autoencoder, rf_model, sde, sampling,
                              **config.get('model', {})).to(device)
    return model


def build_loaders(config):
    data_config = config["data"]
    dataset_specs = data_config.get("datasets")
    if dataset_specs is None:
        dataset_specs = [
            {
                "name": data_config.get("name", "dataset"),
                "infrared_dir": data_config["infrared_dir"],
                "visible_dir": data_config["visible_dir"],
            }
        ]
    dataset_specs = [spec for spec in dataset_specs if spec.get("enabled", True)]
    if not dataset_specs:
        raise ValueError("No enabled Stage-III datasets are configured")
    split_path_keys = {
        "train_infrared_dir",
        "train_visible_dir",
        "val_infrared_dir",
        "val_visible_dir",
    }
    for spec in dataset_specs:
        if not spec.get("partition_by_hash", True):
            missing = sorted(split_path_keys.difference(spec))
            if missing:
                raise ValueError(
                    "Dataset {} disables hash partitioning but lacks: {}".format(
                        spec["name"], ", ".join(missing)
                    )
                )

    def build_split(split):
        parts = []
        audits = []
        for spec in dataset_specs:
            dataset = PairedIRVIDataset(
                dataset_name=spec["name"],
                infrared_dir=spec.get(
                    "{}_infrared_dir".format(split), spec.get("infrared_dir")
                ),
                visible_dir=spec.get(
                    "{}_visible_dir".format(split), spec.get("visible_dir")
                ),
                crop_size=data_config["crop_size"],
                split=split,
                val_fraction=spec.get(
                    "val_fraction", data_config.get("val_fraction", 0.05)
                ),
                seed=config["seed"],
                strict_pairs=spec.get(
                    "strict_pairs", data_config.get("strict_pairs", False)
                ),
                partition_by_hash=spec.get("partition_by_hash", True),
            )
            parts.append(dataset)
            audits.append(dataset.audit)
        combined = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        audit = {
            "split": split,
            "dataset_count": len(parts),
            "selected": sum(item["selected"] for item in audits),
            "paired_total": sum(item["paired_total"] for item in audits),
            "datasets": audits,
        }
        return combined, audit

    train_set, train_audit = build_split("train")
    val_set, val_audit = build_split("val")
    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    loader_common = dict(
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=data_config["num_workers"] > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **loader_common
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_common)
    return train_loader, val_loader, train_audit, val_audit


def gradient_norm(parameters):
    squared = sum(float(p.grad.detach().float().square().sum())
                  for p in parameters if p.grad is not None)
    return math.sqrt(squared)


def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              gradient_clip=1.0, preview_path=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals, steps = {}, 0
    progress = tqdm(loader, desc='train' if training else 'val', leave=False)
    for index, batch in enumerate(progress):
        if max_batches is not None and index >= max_batches:
            break
        visible = batch['visible'].to(device, non_blocking=True)
        infrared = batch['infrared'].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(visible, infrared)
        if not torch.isfinite(output['loss']):
            raise FloatingPointError('Non-finite Stage-III loss at batch {}'.format(index))
        metrics = {key: float(value.detach()) for key, value in output.items()
                   if torch.is_tensor(value) and value.numel() == 1}
        if training:
            output['loss'].backward()
            norm = gradient_norm(model.encoder_parameters())
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError('Encoder gradient is missing or non-finite: {}'.format(norm))
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('A frozen parameter received gradients')
            metrics['encoder_grad_norm'] = norm
            torch.nn.utils.clip_grad_norm_(list(model.encoder_parameters()), gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if preview_path is not None and index == 0:
            with torch.no_grad():
                count = min(4, visible.shape[0])
                panels = torch.stack([_gray(visible[:count]), _gray(infrared[:count]),
                                      _gray(output['fused_image'][:count])], dim=1)
                preview_path = Path(preview_path)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(panels.flatten(0, 1).cpu(), preview_path, nrow=3, padding=2)
        steps += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        if index % 20 == 0:
            progress.set_postfix(loss=totals['loss']/steps,
                                 task=totals['fusion_training_loss']/steps)
    if not steps:
        raise RuntimeError('No batches were processed')
    return {key: value/steps for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                    config, record, best_val, no_improve, loader):
    payload = {
        'format': 'rffusion_stage3', 'epoch': epoch, 'global_step': global_step,
        'encoder': model.autoencoder.encoder.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'config': config, 'metrics': record, 'best_val': best_val, 'no_improve': no_improve,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'loader': loader.generator.get_state()},
    }
    device = next(model.encoder_parameters()).device
    if device.type == 'cuda':
        payload['rng']['cuda'] = torch.cuda.get_rng_state(device)
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_full_vae(path, model, epoch, config, metrics):
    """All VAE parameters, including the unchanged decoder and codebook."""
    payload = {
        'state_dict': {key: value.detach().cpu() for key, value in model.autoencoder.state_dict().items()},
        'stage3_metadata': {'stage3_epoch': int(epoch), 'base_vae': config['paths']['vae_checkpoint'],
                            'metrics': metrics, 'config': copy.deepcopy(config)},
    }
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, loader, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    encoder = payload.get('encoder')
    if encoder is None:
        overlay = payload['stage3']
        if set(overlay) != {'encoder'}:
            raise ValueError('Only encoder-only Stage-III checkpoints can be resumed')
        encoder = overlay['encoder']
    model.autoencoder.encoder.load_state_dict(encoder, strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    rng = payload.get('rng')
    if rng:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        loader.generator.set_state(rng['loader'])
        if device.type == 'cuda' and 'cuda' in rng:
            torch.cuda.set_rng_state(rng['cuda'], device)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/stage3.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--resume-check', action='store_true')
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    if args.resume_check and not args.resume:
        parser.error('--resume-check requires --resume')
    seed_everything(config['seed'])
    device = torch.device(args.device or config['device'])
    train_loader, val_loader, train_audit, val_audit = build_loaders(config)
    model = build_model(config, device)
    train = config['training']
    optimizer = torch.optim.AdamW(model.encoder_parameters(), lr=train['encoder_lr'],
                                  weight_decay=train.get('weight_decay', 0.0))
    total_steps = int(train.get('max_steps') or
                      train.get('scheduler_max_epochs', train['max_epochs'])*len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=train['min_lr'])
    global_step, start_epoch, best_val, no_improve = 0, 1, float('inf'), 0
    if args.resume:
        output_dir = Path(args.resume).resolve().parent
        payload = restore_checkpoint(args.resume, model, optimizer, scheduler, train_loader, device)
        saved = payload['config']
        if saved['data']['batch_size'] != config['data']['batch_size']:
            raise ValueError('Resume batch size must match the saved training run')
        record = payload.get('metrics', {})
        if record.get('steps_in_epoch', len(train_loader)) != len(train_loader):
            raise ValueError('Resume loader must contain the same number of batches')
        start_epoch = int(payload['epoch'])+1
        global_step = int(payload.get('global_step', record.get('global_step', 0)))
        best_val = float(payload.get('best_val', float('inf')))
        no_improve = int(payload.get('no_improve', 0))
        if not math.isfinite(best_val):
            history = output_dir/'metrics.jsonl'
            if history.is_file():
                for line in history.read_text().splitlines():
                    old = json.loads(line)
                    if old.get('val'):
                        best_val = min(best_val, old['val']['fusion_training_loss'])
        print('resumed={} start_epoch={} global_step={}'.format(args.resume, start_epoch, global_step))
        if args.resume_check:
            print('resume_check=passed')
            return
    else:
        output_dir = Path(config['paths']['output_root'])/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir/'dataset_audit.json').write_text(json.dumps({'train':train_audit,'val':val_audit},indent=2))
    print('output_dir={}'.format(output_dir.resolve()), flush=True)
    print('train_audit={} val_audit={}'.format(train_audit, val_audit), flush=True)
    with SummaryWriter(str(output_dir/'tensorboard')) as writer:
        for epoch in range(start_epoch, int(train['max_epochs'])+1):
            if total_steps-global_step < len(train_loader) and not args.dry_run:
                print('stopping_before_partial_epoch')
                break
            train_metrics = run_epoch(model, train_loader, device, optimizer, scheduler,
                                      train.get('gradient_clip',1.0), max_batches=1 if args.dry_run else None)
            global_step += 1 if args.dry_run else len(train_loader)
            validated = epoch % train.get('validate_every_epochs',1) == 0 or epoch == train['max_epochs'] or args.dry_run
            val_metrics = None
            if validated:
                val_metrics = run_epoch(model, val_loader, device,
                    preview_path=output_dir/'validation_samples'/'epoch_{:04d}.png'.format(epoch),
                    max_batches=1 if args.dry_run else None)
            record = {'epoch':epoch,'global_step':global_step,'steps_in_epoch':len(train_loader),
                      'train':train_metrics,'val':val_metrics,'validated':validated,
                      'lr_encoder':optimizer.param_groups[0]['lr']}
            improved = validated and val_metrics['fusion_training_loss'] < best_val-train.get('early_stopping_min_delta',0.0)
            if improved:
                best_val, no_improve = val_metrics['fusion_training_loss'], 0
                save_full_vae(output_dir/'autoencoder_best.ckpt', model, epoch, config, val_metrics)
            elif validated:
                no_improve += 1
            for name in ['last.pth'] + (['best.pth'] if improved else []) + (
                    ['epoch_{:04d}.pth'.format(epoch)] if epoch % train.get('checkpoint_every',1) == 0 else []):
                save_checkpoint(output_dir/name, model, optimizer, scheduler, epoch, global_step,
                                config, record, best_val, no_improve, train_loader)
            with (output_dir/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record,sort_keys=True)+'\n')
            for split, values in [('train',train_metrics),('val',val_metrics)]:
                for key,value in (values or {}).items():
                    writer.add_scalar(split+'/'+key,value,epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
            if args.dry_run or global_step >= total_steps or no_improve >= train.get('early_stopping_patience',6):
                break
    full_best_path = output_dir/'autoencoder_best.ckpt'
    if not full_best_path.is_file():
        # Older runs saved only an encoder overlay. Export their best, not last.
        best_path = next((output_dir/name for name in ('best.pth','stage3_best.pth')
                          if (output_dir/name).is_file()), None)
        if best_path is None:
            raise FileNotFoundError('No validated best checkpoint is available for VAE export')
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        encoder = best.get('encoder')
        if encoder is None:
            encoder = best['stage3']['encoder']
        model.autoencoder.encoder.load_state_dict(encoder, strict=True)
        save_full_vae(full_best_path, model, best['epoch'], config,
                      best.get('metrics', {}).get('val'))
    print('Full best VAE: {}'.format(full_best_path.resolve()))


if __name__ == '__main__':
    main()
