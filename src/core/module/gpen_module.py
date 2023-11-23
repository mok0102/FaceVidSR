import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

import numpy as np

class GPENTrainer(pl.LightningModule):
    def __init__(self, 
                 dataset: dict,
                #  generator: torch.nn.Module,
                #  discriminator: torch.nn.Module,
                #  g_ema: torch.nn.Module,
                 model: torch.nn.Module,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './test')-> None:
        super().__init__()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.g_ema = model.g_ema
        
        
    def step(self, video):
        for i in range(video.shape[1]):
            frame = video[:, i, ::]
            output = self.generator(frame)
        return output
    
    def predict_step(self, batch, batch_idx) -> None:
        degraded_frames, frames = batch
        output = self.step(degraded_frames)
    
    def training_step(self, batch, batch_idx) -> None:
        degraded_frames, frames = batch
        
        self.requires_grad(self.generator, False)
        self.requires_grad(self.discriminator, True)
        
        output = self.step(degraded_frames)
        
        # fake_img, _ = self.generator(degraded_frames)
        # fake_pred = self.discriminator(fake_img)

        # real_pred = self.discriminator(self.real_img)
        # d_loss = self.d_logistic_loss(real_pred, fake_pred)

        gop, dop = self.optimizers()
    
    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def requires_grad(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag
    
    def configure_optimizers(self):
        g_optim = optim.Adam(
            self.generator.parameters(),
            lr=0.002 * 4,
            betas=(0 ** 4, 0.99 ** 4),
        )
        
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=0.002 * 16,
            betas=(0 ** 16, 0.99 ** 16),
        )   
        
        return [g_optim, d_optim], []
        
        