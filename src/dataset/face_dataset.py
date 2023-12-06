import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import numpy as np
import cv2

from typing import Sequence, Dict, Union

import sys
sys.path.append(os.path.realpath('./src'))
from utils.degradation import random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression

class FaceDataset(Dataset):
    def __init__(self, 
                 noise_range: Sequence[float],
                 jpeg_range: Sequence[int],
                 img_path, image_size=1024, scale=2, 
                 transform=None):
        self.img = sorted(glob(os.path.join(img_path, '*/*.png')))
        # self.video = ['/workspace/zlatkd12/mok/zresults/inputs/256/_LOyzaKYIpc_1_0']
        self.image_size = image_size
        self.scale = scale
        self.noise_range=noise_range
        self.jpeg_range=jpeg_range
        
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.img[idx]
        

        img = np.array(Image.open(img_name))
            
        ####### degradation part
        # resize
        lqsize = int(self.image_size // self.scale)
        img_lq = cv2.resize(img, (lqsize, lqsize), interpolation=cv2.INTER_LINEAR)          
        
        # gaussian noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
            
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            
        img_lq = cv2.resize(img_lq, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        
        
        if self.transform:
            img_lq= self.transform(img_lq)
            img = self.transform(img)
            

        # img_lq = torchvision.transforms.ToTensor()(img_lq)
        # img = torchvision.transforms.ToTensor()(img)
        
        img =  (torch.from_numpy(img) - 0.5) / 0.5
        img_lq =  (torch.from_numpy(img_lq) - 0.5) / 0.5
        
        img = img.permute(2, 0, 1)#.flip(0) # BGR->RGB
        img_lq = img_lq.permute(2, 0, 1)#.flip(0) # BGR->RGB
        
        print(img.shape, img_lq.shape)
        
        return img_lq, img
    
    def __len__(self):
        return len(self.img)