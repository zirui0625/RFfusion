import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_A, image_B):
        for t in self.transforms:
            image_A, image_B = t(image_A, image_B)
        return image_A, image_B

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image_A, image_B):
        image_A = F.resize(image_A, self.size)
        image_B = F.resize(image_B, self.size)
        return image_A, image_B
    
class Resize_16(object):
    def __init__(self):
        pass

    def __call__(self, image_A, image_B):
        width, height = image_A.size
        
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        image_A = F.resize(image_A, (new_height, new_width))
        image_B = F.resize(image_B, (new_height, new_width))
        
        return image_A, image_B

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image_A, image_B):
        if random.random() < self.flip_prob:
            image_A = F.hflip(image_A)
            image_B = F.hflip(image_B)
        return image_A, image_B

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image_A, image_B):
        if random.random() < self.flip_prob:
            image_A = F.vflip(image_A)
            image_B = F.vflip(image_B)
        return image_A, image_B

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image_A, image_B):
        image_A = pad_if_smaller(image_A, self.size)
        image_B = pad_if_smaller(image_B, self.size)
        crop_params = T.RandomCrop.get_params(image_A, (self.size, self.size))
        image_A = F.crop(image_A, *crop_params)
        image_B = F.crop(image_B, *crop_params)
        return image_A, image_B

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image_A, image_B):
        image_A = F.center_crop(image_A, self.size)
        image_B = F.center_crop(image_B, self.size)
        return image_A, image_B

class ToTensor(object):
    def __call__(self, image_A, image_B):
        image_A = F.to_tensor(image_A)
        image_B = F.to_tensor(image_B)
        return image_A, image_B

'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_gt, target_gt, image_full):
        for t in self.transforms:
            image, target, image_gt, target_gt, image_full = t(image, target, image_gt, target_gt, image_full)
        return image, target, image_gt, target_gt, image_full


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        image_gt = F.resize(image_gt, self.size, interpolation=T.InterpolationMode.NEAREST)
        target_gt = F.resize(target_gt, self.size, interpolation=T.InterpolationMode.NEAREST)

        return image, target, image_gt, target_gt


class Resize_16(object):
    def __init__(self):
        pass

    def __call__(self, image, target, image_gt, target_gt, image_full):
        width, height = image.size

        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        image = F.resize(image, (new_height, new_width))
        target = F.resize(target, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        image_gt = F.resize(image_gt, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        target_gt = F.resize(target_gt, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        image_full = F.resize(image_full, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)

        return image, target, image_gt, target_gt, image_full


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt, image_full):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            image_gt = F.hflip(image_gt)
            target_gt = F.hflip(target_gt)
            image_full = F.hflip(image_full)
        return image, target, image_gt, target_gt, image_full


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt, image_full):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            image_gt = F.vflip(image_gt)
            target_gt = F.vflip(target_gt)
            image_full = F.vflip(image_full)
        return image, target, image_gt, target_gt, image_full


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt, image_full):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        image_gt = pad_if_smaller(image_gt, self.size)
        target_gt = pad_if_smaller(target_gt, self.size)
        image_full = pad_if_smaller(image_full, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        image_gt = F.crop(image_gt, *crop_params)
        target_gt = F.crop(target_gt, *crop_params)
        image_full = F.crop(image_full, *crop_params)
        return image, target, image_gt, target_gt, image_full

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        image_gt = F.center_crop(image_gt, self.size)
        target_gt = F.center_crop(target_gt, self.size)
        return image, target, image_gt, target_gt


class ToTensor(object):
    def __call__(self, image, target, image_gt, target_gt, image_full):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        image_gt = F.to_tensor(image_gt)
        target_gt = F.to_tensor(target_gt)
        image_full = F.to_tensor(image_full)
        return image, target, image_gt, target_gt, image_full
'''