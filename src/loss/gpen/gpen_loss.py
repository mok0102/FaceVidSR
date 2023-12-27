import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class GPENLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        
    def d_logistic_loss(real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()
    
    def d_r1_loss(real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None):
        smooth_l1_loss, id_loss = loss_funcs
        
        loss = F.softplus(-fake_pred).mean()
        loss_l1 = smooth_l1_loss(fake_img, real_img)
        loss_id, __, __ = id_loss(fake_img, real_img, input_img)
        loss += 1.0*loss_l1 + 1.0*loss_id

        return loss
        
    def forward(self, mel, g):
        return None
    