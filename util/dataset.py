from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class SimpleDataSet(Dataset):
    def __init__(self, data_root, phase='train', transform=None):
        self.phase = phase
        '''
        if phase=='train':
            self.infrared_path = data_root + '/ir/train'
            self.visible_path = data_root + '/vi/train'
        if phase=='eval':
            self.infrared_path = data_root + '/ir/test'
            self.visible_path = data_root + '/vi/test'
        '''
        if phase == 'train':
            self.infrared_path = os.path.join(data_root, 'ir', 'train')
            self.visible_path = os.path.join(data_root, 'vi', 'train')
        elif phase == 'eval':
            self.infrared_path = os.path.join(data_root, 'ir', 'test')
            self.visible_path = os.path.join(data_root, 'vi', 'test')
        else:
            raise ValueError(f"Unknown phase: {phase}")
        self.infrared_files = sorted(os.listdir(self.infrared_path))  
        self.visible_files = sorted(os.listdir(self.visible_path))
        self.transform = transform

    def __len__(self):
        return len(self.infrared_files)

    def __getitem__(self, idx):
        '''
        image_A_path = self.visible_path[item]
        image_B_path = self.infrared_path[item]
        '''
        image_A_path = os.path.join(self.visible_path, self.visible_files[idx])  
        image_B_path = os.path.join(self.infrared_path, self.infrared_files[idx]) 
        
        
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')

        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B = self.transform(image_A, image_B)
            

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, name