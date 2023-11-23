import torch
from torch import optim, nn, utils, Tensor
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

class FaceVidSRTrainer(pl.LightningModule):
    def __init__(self, 
                 dataset: dict,
                 model: torch.nn.Module,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './test')-> None:
        super().__init__()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model = model
        
    def step(self, video):
        output = self.model(video)
        return output
    
    def predict_step(self, batch, batch_idx) -> None:
        frames = batch
        output = self.step(batch)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer