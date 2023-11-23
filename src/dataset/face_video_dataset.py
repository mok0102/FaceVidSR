import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image

class FaceVidDataset(Dataset):
    def __init__(self, video_path, image_size=1024, transform=None):
        self.video = sorted(glob(os.path.join(video_path, '*/')))
        # self.video = ['/workspace/zlatkd12/mok/zresults/inputs/256/_LOyzaKYIpc_1_0']
        self.image_size = image_size
        self.transform = transform
        
    def __getitem__(self, idx):
        video_name = self.video[idx]
        
        frame_paths = sorted(glob(os.path.join(video_name, '*.png')))
        
        frame = torch.zeros(len(frame_paths), 3, self.image_size, self.image_size)
    
        for i, v in enumerate(frame_paths):
            img = Image.open(v)
            img = torchvision.transforms.ToTensor()(img)
            
            frame[i] = img
            
        if self.transform:
            frame = self.transform(frame)
            
        return frame
    
    def __len__(self):
        return len(self.video)