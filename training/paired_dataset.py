"""Paired IR/VI image dataset with geometry-consistent random crops."""

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IMAGE_INDEX_CACHE: Dict[str, Dict[str, Path]] = {}


def _ir_valid_region(infrared: torch.Tensor) -> np.ndarray:
    """Estimate NII's warp support, keeping enclosed dark objects valid.

    Only exact-zero regions connected to the image boundary are padding.
    A one-native-pixel guard excludes the immediate padding/JPEG boundary.
    This is not a brightness threshold or an official registration mask.
    """
    from scipy.ndimage import binary_dilation, binary_propagation

    zero = infrared[0].numpy() == 0
    boundary = np.zeros_like(zero)
    boundary[0, :] = zero[0, :]
    boundary[-1, :] = zero[-1, :]
    boundary[:, 0] = zero[:, 0]
    boundary[:, -1] = zero[:, -1]
    padding = binary_propagation(boundary, mask=zero)
    return ~binary_dilation(padding, iterations=1)


def _bilinear_source_bounds(start, stop, source_size, target_size):
    """Native pixels contributing to a target interval, align_corners=False."""
    lower = math.floor((start + 0.5) * source_size / target_size - 0.5)
    upper = math.ceil((stop - 0.5) * source_size / target_size - 0.5) + 1
    return max(0, lower), min(source_size, upper)


def _index_images(root: str) -> Dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError("Image directory not found: {}".format(root_path))
    cache_key = str(root_path)
    if cache_key in _IMAGE_INDEX_CACHE:
        return _IMAGE_INDEX_CACHE[cache_key]
    indexed = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root_path)
        key = str(relative.with_suffix("")).replace("\\", "/")
        if key in indexed:
            raise ValueError("Duplicate pairing key {!r} under {}".format(key, root_path))
        indexed[key] = path
    _IMAGE_INDEX_CACHE[cache_key] = indexed
    return indexed


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2 ** 64)


class PairedIRVIDataset(Dataset):
    def __init__(
        self,
        infrared_dir: str,
        visible_dir: str,
        dataset_name: str = "dataset",
        crop_size: Union[int, Tuple[int, int]] = 256,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 23,
        strict_pairs: bool = False,
        partition_by_hash: bool = True,
    ):
        super().__init__()
        self.dataset_name = str(dataset_name)
        if isinstance(crop_size, (list, tuple)):
            if len(crop_size) != 2:
                raise ValueError("crop_size sequence must be (height, width)")
            crop_height, crop_width = (int(value) for value in crop_size)
        else:
            crop_height = crop_width = int(crop_size)
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("crop_size dimensions must be positive")
        self.crop_size = (crop_height, crop_width)
        self.split = split.lower()
        self.seed = int(seed)
        self.partition_by_hash = bool(partition_by_hash)
        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0.0 < float(val_fraction) < 1.0:
            raise ValueError("val_fraction must lie in (0, 1)")

        infrared = _index_images(infrared_dir)
        visible = _index_images(visible_dir)
        ir_only = sorted(set(infrared).difference(visible))
        vi_only = sorted(set(visible).difference(infrared))
        if strict_pairs and (ir_only or vi_only):
            raise ValueError(
                "Unpaired files: {} IR-only and {} VI-only".format(
                    len(ir_only), len(vi_only)
                )
            )
        paired_keys = sorted(set(infrared).intersection(visible))
        selected = []
        for key in paired_keys:
            if not self.partition_by_hash:
                selected.append((key, infrared[key], visible[key]))
                continue
            split_key = "{}/{}/{}".format(self.seed, self.dataset_name, key)
            is_val = _stable_fraction(split_key) < float(val_fraction)
            if (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                selected.append((key, infrared[key], visible[key]))
        if not selected:
            raise ValueError("No paired images selected for split {!r}".format(self.split))
        self.samples: List[Tuple[str, Path, Path]] = selected
        self.audit = {
            "dataset": self.dataset_name,
            "ir_total": len(infrared),
            "vi_total": len(visible),
            "paired_total": len(paired_keys),
            "ir_only": len(ir_only),
            "vi_only": len(vi_only),
            "selected": len(selected),
            "split": self.split,
            "partition_by_hash": self.partition_by_hash,
        }
        if self.dataset_name == "NII_CU_RGB_T":
            self.audit["crop_region"] = "exclude_boundary_connected_ir_padding"

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load(path: Path, mode: str) -> torch.Tensor:
        with Image.open(str(path)) as image:
            image = image.convert(mode)
            array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _paired_crop(self, visible: torch.Tensor, infrared: torch.Tensor, index: int):
        valid_region = (
            _ir_valid_region(infrared)
            if self.dataset_name == "NII_CU_RGB_T" else None
        )
        intermediate_size = visible.shape[-2:]
        if visible.shape[-2:] != infrared.shape[-2:]:
            infrared = F.interpolate(
                infrared.unsqueeze(0),
                size=visible.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        height, width = visible.shape[-2:]
        crop_height, crop_width = self.crop_size
        scale = max(
            crop_height / float(height), crop_width / float(width), 1.0
        )
        if scale > 1.0:
            new_size = (int(round(height * scale)), int(round(width * scale)))
            visible = F.interpolate(
                visible.unsqueeze(0), new_size, mode="bicubic", align_corners=False
            ).squeeze(0)
            infrared = F.interpolate(
                infrared.unsqueeze(0), new_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            height, width = new_size

        if valid_region is not None:
            top, left = self._valid_crop_origin(
                valid_region, intermediate_size, (height, width), index
            )
            visible = visible[:, top : top + crop_height, left : left + crop_width]
            infrared = infrared[:, top : top + crop_height, left : left + crop_width]
            if self.split == "train" and bool(torch.rand(()) < 0.5):
                visible = torch.flip(visible, dims=[2])
                infrared = torch.flip(infrared, dims=[2])
            return visible, infrared

        if self.split == "train":
            top = int(torch.randint(0, height - crop_height + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_width + 1, (1,)).item())
            if bool(torch.rand(()) < 0.5):
                visible = torch.flip(visible, dims=[2])
                infrared = torch.flip(infrared, dims=[2])
        else:
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
        visible = visible[:, top : top + crop_height, left : left + crop_width]
        infrared = infrared[:, top : top + crop_height, left : left + crop_width]
        return visible, infrared

    def _valid_crop_origin(self, valid, intermediate_size, output_size, index):
        crop_height, crop_width = self.crop_size
        height, width = output_size

        def usable(top, left):
            intervals = []
            for start, length, native, middle, final in zip(
                (top, left), self.crop_size, valid.shape, intermediate_size, output_size
            ):
                lower, upper = _bilinear_source_bounds(start, start + length, middle, final)
                intervals.append(_bilinear_source_bounds(lower, upper, native, middle))
            (top_native, bottom_native), (left_native, right_native) = intervals
            return valid[top_native:bottom_native, left_native:right_native].all()

        center = ((height - crop_height) // 2, (width - crop_width) // 2)
        if self.split == "val" and usable(*center):
            return center
        generator = None
        if self.split == "val":
            generator = torch.Generator().manual_seed(self.seed + index)
        for _ in range(32):
            top = int(torch.randint(height - crop_height + 1, (1,), generator=generator).item())
            left = int(torch.randint(width - crop_width + 1, (1,), generator=generator).item())
            if usable(top, left):
                return top, left
        if usable(*center):
            return center
        raise ValueError(
            "No padding-free {} crop found for {} sample index {}; "
            "check the source image's valid region".format(self.crop_size, self.dataset_name, index)
        )

    def __getitem__(self, index):
        key, infrared_path, visible_path = self.samples[index]
        visible = self._load(visible_path, "RGB")
        infrared = self._load(infrared_path, "L")
        visible, infrared = self._paired_crop(visible, infrared, index)
        return {
            "visible": visible,
            "infrared": infrared,
            "name": self.dataset_name + "__" + key.replace("/", "__"),
            "dataset": self.dataset_name,
            "visible_path": str(visible_path),
            "infrared_path": str(infrared_path),
        }
