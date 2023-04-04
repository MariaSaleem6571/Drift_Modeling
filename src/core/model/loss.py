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
    

class ProbDistrLoss():
    def __init__(self, exclude_mask, batch_mean=False, error_type="MAE", alpha=0.3):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean
        self._error_type = error_type
        self._alpha = alpha

    def __call__(self, pred, true,  *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        if self._error_type == "MAE":
            error = torch.abs(pred - true).mean(1)
        elif self._error_type == "MSE":
            error = torch.square(pred - true).mean(1)
            
        error = (1 - self._alpha)*error + self._alpha*torch.abs(1 - torch.sum(pred))
        
        if self._batch_mean:
            error = error.mean()
        return error