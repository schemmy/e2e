import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QauntileLoss','QauntileLossNew']

class QauntileLoss(nn.Module):
    def __init__(self, quantile=0.1, size_average=True):
        super(QauntileLoss, self).__init__()
        self.quantile = quantile
        self.size_average = size_average
    
    # score is N x 1
    def forward(self, input, target):
        
        #print(input.size(),target.size())
        diff = input - target
        zero_1 = torch.zeros_like(diff).cuda()
        zero_2 = torch.zeros_like(diff).cuda()
        loss = self.quantile * torch.max(-diff,zero_1) + (1-self.quantile) * torch.max(diff,zero_2)
        
        return loss.mean() if self.size_average else loss.sum()
        
        

class QauntileLossNew(nn.Module):
    def __init__(self, y_norm=True, size_average=True, use_square=True):
        super(QauntileLossNew, self).__init__()
        #self.quantile = quantile
        self.size_average = size_average
        self.y_norm = y_norm
        self.use_square = use_square
    
    # score is N x 31
    def forward(self, input, target, quantile=0.85):
        
        if self.use_square:
            input = torch.pow(input, 2)
        
        #print(input.size(),target.size())
        diff = input - target
        zero_1 = torch.zeros_like(diff)#.cuda()
        zero_2 = torch.zeros_like(diff)#.cuda()
        loss = quantile * torch.max(-diff,zero_1) + (1-quantile) * torch.max(diff,zero_2)
        
        #print(target.size(), target.sum())
        if self.y_norm:
            loss /= max(1.0, target.sum())
        
        return loss.mean() if self.size_average else loss.sum()
        