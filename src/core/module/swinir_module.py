import torch
from torch import optim, nn, utils, Tensor
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

from einops import rearrange

class SwinIRTrainer(pl.LightningModule):
    def __init__(self, 
                 dataset: dict,
                 model: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: torch.nn.Module,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './test')-> None:
        super().__init__()
        self.dataset = dataset
        print(len(self.dataset))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
        self.model = model        
        self.criterion = loss
        self.optimizer = optimizer
    
    def train_dataloader(self):
        return self.dataloader
    
    def val_dataloader(self):
        return self.dataloader
    def step(self, video):
        output = self.model(video)
        return output
    
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    
    def predict_step(self, batch, batch_idx) -> None:
        frames = batch
        output = self.step(batch)
        
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
    
    def step(self, img) -> torch.Tensor:
        return self.model(img)
        
    def training_step(self, batch, batch_idx: int):                
        lq, hq = batch #b c h w
        pred = self.step(lq)
        loss = self.criterion(pred, hq)
        self.log("train_loss", loss, on_step=True)
        return loss
                
        
    def predict_step(self, batch, batch_idx: int):
        lq, hq = batch
        return self.step(lq)
        
        
        