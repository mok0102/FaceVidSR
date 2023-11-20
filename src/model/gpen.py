import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import sys
# import os
# sys.path.append(os.path.realpath("./"))
# import cv2

class GPEN(nn.Module):
    def __init__(self, device):
        super(GPEN, self).__init__()
        self.device = device
        
    def forward(self, video):
        return video