import torch
from torch import optim, nn, utils, Tensor
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

class CodeFormerTrainer(pl.LightningModule):
    def __init__(self, 
                 dataset: dict,
                 model: torch.nn.Module,
                 elsemodel: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: torch.nn.Module,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './test',
                 w=0)-> None:
        super().__init__()
        self.dataset = dataset
        print(len(self.dataset))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
        
        self.model = model      
        self.vqgan = elsemodel
        self.vqgan.eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False
                  
        self.cross_entropy = loss
        self.optimizer = optimizer
        
        self.w = w #### if w=1 stage3
        self.feat_loss_weight = 1
        self.entropy_loss_weight = 0.5
    
    
    def train_dataloader(self):
        return self.dataloader
    
    def val_dataloader(self):
        return self.dataloader
    def step(self, video):
        output = self.model(video)
        return output
    
    def predict_step(self, batch, batch_idx) -> None:
        lq, hq = batch
        
        output = self.step(batch)
        
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
    
    def step(self, img) -> torch.Tensor:
        return self.model(img, w=self.w, code_only=True)
        
    def training_step(self, batch, batch_idx: int):                
        lq, hq = batch #b c h w
        b = lq.shape[0]
        
        x = self.vqgan.encoder(hq)
        _, _, quant_stats = self.vqgan.quantize(x)
        min_encoding_indices = quant_stats['min_encoding_indices']
        idx_gt = min_encoding_indices.view(b, -1)
        
        quant_feat_gt = self.model.quantize.get_codebook_feat(idx_gt, shape=[b,16,16,256])
        logits, lq_feat = self.step(lq)
        l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
        
        
        ce_loss = self.cross_entropy(logits.permute(0, 2, 1), idx_gt)
        
        loss = l_feat_encoder + ce_loss
        self.log("train_loss", loss, on_step=True)
        return loss
                
        
    def predict_step(self, batch, batch_idx: int):
        lq, hq = batch
        return self.step(lq)
        
        
        