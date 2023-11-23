import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
import os
sys.path.append(os.path.realpath("./src/external/GPEN"))

from face_model.gpen_model import FullGenerator, Discriminator

class GPEN(nn.Module):
    def __init__(self, save_path, device):
        super(GPEN, self).__init__()
        self.save_path = save_path
        self.device = device
        
    #     self.generator = FullGenerator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    # ).to(self.device)
        
        self.generator = FullGenerator(256, 512, 8, channel_multiplier=2, narrow=1.0, device=device).to(self.device)
        self.discriminator = Discriminator(
        256, channel_multiplier=2, narrow=1.0, device=device
    ).to(self.device)
        
        self.g_ema = FullGenerator(
        256, 512, 8, channel_multiplier=2, narrow=1.0, device=device
    ).to(self.device)
        
        
    def forward(self, video):
        return video