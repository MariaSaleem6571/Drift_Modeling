import torch
from torch import nn


class MSE():
    def __init__(self, exclude_mask, batch_mean=False):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean

    def __call__(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        error = torch.square(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error
    
    
class MAE():
    def __init__(self, exclude_mask, batch_mean=False):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean

    def __call__(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        error = torch.abs(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error


class ResidualLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self._loss_fn = loss_fn
    
    def forward(self, pred, true, inp):
        return self._loss_fn(pred, (true-inp))
    

class MAEProbDistrLoss():
    def __init__(self, exclude_mask, batch_mean=False, alpha=0.3):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean
        self._alpha = alpha

    def __call__(self, pred, true,  *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        reg_error = torch.abs(pred - true).mean(1)
        error = (1 - self._alpha)*reg_error + self._alpha*torch.abs(1 - torch.sum(pred))
        
        if self._batch_mean:
            error = error.mean()
        return error


class MSEProbDistrLoss():
    def __init__(self, exclude_mask, batch_mean=False, alpha=0.2):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean
        self._alpha = alpha

    def __call__(self, pred, true,  *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        reg_error = torch.square(pred - true).mean(1)
        error = (1 - self._alpha) * reg_error + self._alpha * torch.abs(1 - torch.sum(pred))
        
        if self._batch_mean:
            error = error.mean()
        return error
