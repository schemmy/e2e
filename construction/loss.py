# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-09 16:37:29


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle     
from sklearn.model_selection import train_test_split
from config import *

class E2E_loss(nn.Module):
    def __init__(self, size_average=True):
        super(E2E_loss, self).__init__()

        self.size_average = size_average
        self.lmd_1 = 0.5
        self.lmd_2 = 0.5

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target, qtl=0.9):

        diff = out - target

        zero_1 = torch.zeros_like(diff)#.cuda()
        zero_2 = torch.zeros_like(diff)#.cuda()
        qtl_loss = qtl * torch.max(-diff, zero_1) + (1 - qtl) * torch.max(diff, zero_2)
        self.qtl_loss = qtl_loss.mean() if self.size_average else qtl_loss.sum()

        # self.vlt_loss = nn.MSELoss()(vlt_o, vlt_t)
        # self.sf_loss = nn.MSELoss()(sf_o, sf_t)
        
        # self.loss = self.qtl_loss + self.lmd_1 * self.vlt_loss + self.lmd_2 * self.sf_loss

        return self.qtl_loss
        


class E2E_Dataset(data.Dataset):

    def __init__(self, phase, data_dir='../data/'):

        self.data_dir = data_dir
        self.phase = phase

        with open(self.data_dir+'1320_feature/df_e2e.pkl', 'rb') as fp:
            o2 = pickle.load(fp)


        sku_set = o2.sku_id.unique()
        sku_train, sku_test = train_test_split(sku_set, random_state=10, train_size=0.8, test_size=0.2)

        self.train_indices = list(o2[o2['sku_id'].isin(sku_train)].index)
        self.test_indices = list(o2[o2['sku_id'].isin(sku_test)].index)

        self.train_len = len(self.train_indices)
        self.test_len = len(self.test_indices)

        self.df = o2[SCALE_FEA+LABEL].astype(float)
        print(self.train_len, self.test_len)
                    

    def __getitem__(self, idx):

        
        if self.phase=='test':
            index = self.test_indices[idx]
        else:
            index = self.train_indices[idx]

        X = self.df.iloc[index]

        a = torch.from_numpy(X[VLT_FEA].values).float()
        b = torch.from_numpy(X[SF_FEA].values).float()
        c = torch.from_numpy(X[CAT_FEA_HOT].values).float()
        d = torch.from_numpy(X[MORE_FEA].values).float()
        e = torch.from_numpy(X[IS_FEA].values).float()
        f = torch.from_numpy(X[LABEL].values).float()
        g = torch.from_numpy(X[LABEL_vlt].values).float()
        h = torch.from_numpy(X[LABEL_sf].values).float()
        return (a,b,c,d,e,f,g,h)
        
    def __len__(self):

        if self.phase=='train':
            return self.train_len
        else:
            return self.test_len



def get_loader(phase, batch_size, shuffle, num_workers):

    dset = E2E_Dataset(phase)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True
                                              )
    return data_loader