import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image

import sys
sys.path.append(os.path.realpath('./src'))
from utils.degradation import random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression

class FaceVidDataset(Dataset):
    def __init__(self, video_path, image_size=1024, scale =2, transform=None):
        self.video = sorted(glob(os.path.join(video_path, '*/')))
        # self.video = ['/workspace/zlatkd12/mok/zresults/inputs/256/_LOyzaKYIpc_1_0']
        self.image_size = image_size
        self.scale = scale
        self.transform = transform
        
    def __getitem__(self, idx):
        video_name = self.video[idx]
        
        frame_paths = sorted(glob(os.path.join(video_name, '*.png')))
        
        frame = torch.zeros(len(frame_paths), 3, self.image_size, self.image_size)
    
        for i, v in enumerate(frame_paths):
            img = Image.open(v)
            img = torchvision.transforms.ToTensor()(img)
            
            frame[i] = img
            
            
        ####### degradation part
        # scale
        lqsize = self.image_size // self.scale
        lqframe = torch.zeros(len(frame_paths), 3, lqsize, lqsize)
        for i in range(frame.shape[0]):
            img = frame[i, ::].copy()
            img = F.interpolate(img, lqsize)
            
            lqframe[i] = img
            
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            
        
        if self.transform:
            lqframe= self.transform(lqframe)
            frame = self.transform(frame)
            
            
        return frame
    
    def __len__(self):
        return len(self.video)