import torch
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
        self.dataloader = DataLoader(dataset, )
        self.model = model
        
    def step(self, video):
        output = self.model(video)
        return output
    
    def predict_step(self, batch, batch_idx) -> None:
        frames = batch
        output = self.step(batch)
        
        
        