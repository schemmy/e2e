# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-10 10:16:51


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

    def __init__(self, phase, gpu, device, data_dir='../data/'):

        self.data_dir = data_dir
        self.phase = phase
        self.gpu = gpu
        self.device = device

        with open(self.data_dir+'1320_feature/df_e2e.pkl', 'rb') as fp:
            o2 = pickle.load(fp)


        sku_set = o2.sku_id.unique()
        sku_train, sku_test = train_test_split(sku_set, random_state=10, train_size=0.8, test_size=0.2)

        self.train_indices = list(o2[o2['sku_id'].isin(sku_train)].index)
        self.test_indices = list(o2[o2['sku_id'].isin(sku_test)].index)

        self.train_len = len(self.train_indices)
        self.test_len = len(self.test_indices)

        self.df = o2[SCALE_FEA+LABEL].astype(float)

        self.df_vlt = self.df[VLT_FEA].values
        self.df_sf = self.df[SF_FEA].values
        self.df_cat = self.df[CAT_FEA_HOT].values
        self.df_oth = self.df[MORE_FEA].values
        self.df_is = self.df[IS_FEA].values
        self.df_label = self.df[LABEL].values
        self.df_label_vlt = self.df[LABEL_vlt].values
        self.df_label_sf = self.df[LABEL_sf].values

        print(self.train_len, self.test_len)
                    

    def __getitem__(self, idx):

        if self.phase=='test':
            index = self.test_indices[idx]
        else:
            index = self.train_indices[idx]

        # Need optimize
        if self.gpu:
            a = torch.from_numpy(self.df_vlt[index]).float().to(self.device)
            b = torch.from_numpy(self.df_sf[index]).float().to(self.device)
            c = torch.from_numpy(self.df_cat[index]).float().to(self.device)
            d = torch.from_numpy(self.df_oth[index]).float().to(self.device)
            e = torch.from_numpy(self.df_is[index]).float().to(self.device)
            f = torch.from_numpy(self.df_label[index]).float().to(self.device)
            g = torch.from_numpy(self.df_label_vlt[index]).float().to(self.device)
            h = torch.from_numpy(self.df_label_sf[index]).float().to(self.device)
        else:
            a = torch.from_numpy(self.df_vlt[index]).float()
            b = torch.from_numpy(self.df_sf[index]).float()
            c = torch.from_numpy(self.df_cat[index]).float()
            d = torch.from_numpy(self.df_oth[index]).float()
            e = torch.from_numpy(self.df_is[index]).float()
            f = torch.from_numpy(self.df_label[index]).float()
            g = torch.from_numpy(self.df_label_vlt[index]).float()
            h = torch.from_numpy(self.df_label_sf[index]).float()

        return (a,b,c,d,e,f,g,h)
        

    def __len__(self):

        if self.phase=='train':
            return self.train_len
        else:
            return self.test_len



def get_loader(phase, batch_size, gpu, device, shuffle, num_workers=0):

    dset = E2E_Dataset(phase, gpu, device)
    # if phase == 'test':
        # batch_size = dset.test_len
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=0,
                                              # pin_memory=True
                                              )
    return data_loader