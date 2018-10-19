# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-19 15:02:00


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle  
import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from configs.config import *
import datetime as dt




class QauntileLoss(nn.Module):
    def __init__(self, y_norm=True, size_average=True, use_square=True):
        super(QauntileLoss, self).__init__()
        #self.quantile = quantile
        self.size_average = size_average
        self.y_norm = y_norm
        self.use_square = use_square
    
    # score is N x 31
    def forward(self, input, target, quantile=0.7):
        
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


class E2E_loss(nn.Module):
    def __init__(self, device, size_average=True):
        super(E2E_loss, self).__init__()

        self.device = device
        self.size_average = size_average
        self.lmd_1 = 0.5
        self.lmd_2 = 0.5

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target, qtl=0.7):

        diff = out - target
        zero_1 = torch.zeros_like(diff).to(self.device)
        zero_2 = torch.zeros_like(diff).to(self.device)
        qtl_loss = qtl * torch.max(-diff, zero_1) + (1 - qtl) * torch.max(diff, zero_2)
        self.qtl_loss = qtl_loss.mean() if self.size_average else qtl_loss.sum()

        self.vlt_loss = nn.MSELoss()(vlt_o, vlt_t)

        diff_sf = sf_o - sf_t
        sf_loss = 0.9 * torch.max(-diff_sf, zero_1) + 0.1 * torch.max(diff_sf, zero_2)
        self.sf_loss = sf_loss.mean() if self.size_average else sf_loss.sum()
        
        self.loss = self.qtl_loss + self.lmd_1 * self.vlt_loss + self.lmd_2 * self.sf_loss

        return self.qtl_loss, self.loss
        


class E2E_v6_loss(nn.Module):
    def __init__(self, device, size_average=True):
        super(E2E_v6_loss, self).__init__()

        self.device = device
        self.size_average = size_average
        self.lmd_1 = 0.5
        self.lmd_2 = 0.05
        self.quantile_loss = QauntileLoss(y_norm=False, size_average=True, use_square=False)

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target, qtl=0.7):

        diff = out - target

        zero_1 = torch.zeros_like(diff).to(self.device)
        zero_2 = torch.zeros_like(diff).to(self.device)
        qtl_loss = qtl * torch.max(-diff, zero_1) + (1 - qtl) * torch.max(diff, zero_2)
        self.qtl_loss = qtl_loss.mean() if self.size_average else qtl_loss.sum()

        self.vlt_loss = nn.MSELoss()(vlt_o, vlt_t)

        sf_o = sf_o.view(-1, rnn_pred_long, num_quantiles)
        self.sf_loss = torch.tensor(0.0).to(self.device)
        for k in range(num_quantiles):
            self.sf_loss += self.quantile_loss(sf_o[:,:,k], sf_t, quantiles[k])
        self.sf_loss /= num_quantiles

        self.loss = self.qtl_loss + self.lmd_1 * self.vlt_loss + self.lmd_2 * self.sf_loss

        return self.qtl_loss, self.loss


  