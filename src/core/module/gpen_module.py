import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

import numpy as np

import sys
sys.path.append('../../external/GPEN')
from training.loss.id_loss import IDLoss

class GPENTrainer(pl.LightningModule):
    def __init__(self, 
                 dataset: dict,
                 model: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer_g: torch.nn.Module,
                 optimizer_d: torch.nn.Module,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './test')-> None:
        super().__init__()
        self.dataset = dataset
        print(len(self.dataset))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
        self.model = model
        
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.g_ema = model.g_ema
        
        # self.idloss = loss
        
        self.automatic_optimization = False
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        self.id_loss = IDLoss(base_dir='/workspace/zlatkd12/mok/FaceVidSR/src/external/GPEN/')
        
        
        
    
    def train_dataloader(self):
        return self.dataloader
    
    def val_dataloader(self):
        return self.dataloader
    def step(self, video):
        output = self.model(video)
        return output
    
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    
    def predict_step(self, batch, batch_idx) -> None:
        frames = batch
        output = self.step(batch)
        
    def configure_optimizers(self):
        return [self.optimizer_g, self.optimizer_d], []
    
    def step(self, img) -> torch.Tensor:
        return self.generator(img)
    
    ###### loss in here (later should be moved)
    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred, fake_img=None, real_img=None, input_img=None):
        # smooth_l1_loss = self.smooth_l1_loss
        # id_loss = self.id_loss
        # id_loss = IDLoss(base_dir='/workspace/zlatkd12/mok/FaceVidSR/src/external/GPEN/')
        loss = F.softplus(-fake_pred).mean()
        loss_l1 = self.smooth_l1_loss(fake_img, real_img)
        loss_id, __, __ = self.id_loss(fake_img, real_img, input_img)
        loss += 1.0*loss_l1 + 1.0*loss_id

        return loss
    ######
        
    def training_step(self, batch, batch_idx: int):
        degraded_img, real_img = batch
        
        optim_g, optim_d = self.optimizers()
        
        #### train discriminator
        self.requires_grad(self.generator, False)
        self.requires_grad(self.discriminator, True)
        
        fake_img, _ = self.generator(degraded_img)
        fake_pred = self.discriminator(fake_img)
        
        real_pred = self.discriminator(real_img)
        # d_loss = d_logistic_loss(real_pred, fake_pred)
        d_loss = self.d_logistic_loss(real_pred, fake_pred)
        
        self.discriminator.zero_grad()
        d_loss.backward()
        optim_d.step()
    
        # d_reg = batch_idx % 16 == 0
        # if d_reg:
        #     real_img.requires_grad = True
        #     real_pred = self.discriminator(real_img)
        #     r1_loss = self.d_r1_loss(real_pred, real_img)
        #     self.discriminator.zero_grad()
        #     (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
        #     optim_d.step()
        
        
        #### train generator
        self.requires_grad(self.generator, True)
        self.requires_grad(self.discriminator, False)
        
        fake_img, _ = self.generator(degraded_img)
        fake_pred = self.discriminator(fake_img)
        g_loss = self.g_nonsaturating_loss(fake_pred, fake_img, real_img, degraded_img)
        
        self.generator.zero_grad()
        g_loss.backward()
        optim_g.step()
    
        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})
        # return self.log_dict
        
    def predict_step(self, batch, batch_idx: int):
        degraded_img, real_img = batch
        fake_img, _ = self.generator(degraded_img)
        
        return fake_img
        