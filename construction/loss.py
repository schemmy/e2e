# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-15 15:57:51


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle  
import pandas as pd   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import *




class QauntileLoss(nn.Module):
    def __init__(self, y_norm=True, size_average=True, use_square=True):
        super(QauntileLoss, self).__init__()
        #self.quantile = quantile
        self.size_average = size_average
        self.y_norm = y_norm
        self.use_square = use_square
    
    # score is N x 31
    def forward(self, input, target, quantile=0.8):
        
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
    def __init__(self, size_average=True):
        super(E2E_loss, self).__init__()

        self.size_average = size_average
        self.lmd_1 = 0.5
        self.lmd_2 = 0.5

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target, qtl=0.8):

        diff = out - target

        zero_1 = torch.zeros_like(diff)#.cuda()
        zero_2 = torch.zeros_like(diff)#.cuda()
        qtl_loss = qtl * torch.max(-diff, zero_1) + (1 - qtl) * torch.max(diff, zero_2)
        self.qtl_loss = qtl_loss.mean() if self.size_average else qtl_loss.sum()

        self.vlt_loss = nn.MSELoss()(vlt_o, vlt_t)
        self.sf_loss = nn.MSELoss()(sf_o, sf_t)
        
        self.loss = self.qtl_loss + self.lmd_1 * self.vlt_loss + self.lmd_2 * self.sf_loss

        return self.qtl_loss, self.loss
        


class E2E_v6_loss(nn.Module):
    def __init__(self, device, size_average=True):
        super(E2E_v6_loss, self).__init__()

        self.device = device
        self.size_average = size_average
        self.lmd_1 = 0.5
        self.lmd_2 = 0.5
        self.quantile_loss = QauntileLoss(y_norm=False, size_average=True, use_square=False)

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target, qtl=0.8):

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



class E2E_Dataset(data.Dataset):

    def __init__(self, o2, sku_train, sku_test, phase, device):

        self.phase = phase
        self.device = device

        self.train_indices = list(o2[o2['sku_id'].isin(sku_train)].index)
        self.test_indices = list(o2[o2['sku_id'].isin(sku_test)].index)

        self.train_len = len(self.train_indices)
        self.test_len = len(self.test_indices)

        df_train = o2[o2['sku_id'].isin(sku_train)]
        df_test = o2[o2['sku_id'].isin(sku_test)]

        X_train_ns, y_train_ns, id_train = df_train[SCALE_FEA], df_train[LABEL], df_train[IDX]
        X_test_ns, y_test_ns, id_test = df_test[SCALE_FEA], df_test[LABEL], df_test[IDX]

        self.n_train, self.n_test = len(X_train_ns), len(X_test_ns)
        self.X_scaler = MinMaxScaler() # For normalizing dataset
        self.y_scaler = MinMaxScaler() # For normalizing dataset
        # We want to predict Close value of stock 
        self.X_train = pd.DataFrame(self.X_scaler.fit_transform(X_train_ns), columns=X_train_ns.columns)
        self.y_train = pd.DataFrame(self.y_scaler.fit_transform(y_train_ns), columns=y_train_ns.columns)

        self.X_test = pd.DataFrame(self.X_scaler.transform(X_test_ns), columns=X_test_ns.columns)
        self.y_test = pd.DataFrame(self.y_scaler.transform(y_test_ns), columns=y_test_ns.columns)

        # pd_scaler = pd.concat([pd.DataFrame([self.y_scaler.data_min_, self.y_scaler.scale_], columns=y_train_ns.columns),
        #             pd.DataFrame([self.X_scaler.data_min_, self.X_scaler.scale_], columns=X_train_ns.columns)], axis=1)
        # pd_scaler.to_csv(self.data_dir+'1320_feature/scaler.csv', index=False)

        if self.phase=='test':
            df = self.X_test.astype(float)
            lb = self.y_test.astype(float)
        else:
            df = self.X_train.astype(float)
            lb = self.y_train.astype(float)

        self.df_vlt = df[VLT_FEA].values
        self.df_sf = df[SF_FEA].values
        self.df_cat = df[CAT_FEA_HOT].values
        self.df_oth = df[MORE_FEA].values
        self.df_is = df[IS_FEA].values
        self.df_label = lb.values
        self.df_label_vlt = df[LABEL_vlt].values
        self.df_label_sf = df[LABEL_sf].values

        print(self.train_len, self.test_len)
                    

    def __getitem__(self, idx):

        # if self.phase=='test':
        #     index = self.test_indices[idx]
        # else:
        #     index = self.train_indices[idx]
        index = idx

        # Need optimize
        if self.device != 'cpu':
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






class E2E_v6_Dataset(data.Dataset):

    def __init__(self, o2, sku_train, sku_test, phase, device):

        self.phase = phase
        self.device = device

        self.train_indices = list(o2[o2['sku_id'].isin(sku_train)].index)
        self.test_indices = list(o2[o2['sku_id'].isin(sku_test)].index)

        self.train_len = len(self.train_indices)
        self.test_len = len(self.test_indices)

        df_train = o2[o2['sku_id'].isin(sku_train) ]
        df_test = o2[o2['sku_id'].isin(sku_test)]

        X_train_ns, y_train_ns, id_train = df_train[SCALE_FEA], df_train[LABEL], df_train[IDX]
        X_test_ns, y_test_ns, id_test = df_test[SCALE_FEA], df_test[LABEL], df_test[IDX]

        self.n_train, self.n_test = len(X_train_ns), len(X_test_ns)
        self.X_scaler = MinMaxScaler() # For normalizing dataset
        self.y_scaler = MinMaxScaler() # For normalizing dataset
        # We want to predict Close value of stock 
        self.X_train = pd.DataFrame(self.X_scaler.fit_transform(X_train_ns), columns=X_train_ns.columns)
        self.y_train = pd.DataFrame(self.y_scaler.fit_transform(y_train_ns), columns=y_train_ns.columns)

        self.X_test = pd.DataFrame(self.X_scaler.transform(X_test_ns), columns=X_test_ns.columns)
        self.y_test = pd.DataFrame(self.y_scaler.transform(y_test_ns), columns=y_test_ns.columns)

        # pd_scaler = pd.concat([pd.DataFrame([self.y_scaler.data_min_, self.y_scaler.scale_], columns=y_train_ns.columns),
        #             pd.DataFrame([self.X_scaler.data_min_, self.X_scaler.scale_], columns=X_train_ns.columns)], axis=1)
        # pd_scaler.to_csv(self.data_dir+'1320_feature/scaler.csv', index=False)

        if self.phase=='test':
            df = self.X_test.astype(float)
            lb = self.y_test.astype(float)
            s2s = df_test
        else:
            df = self.X_train.astype(float)
            lb = self.y_train.astype(float)
            s2s = df_train

        self.df_vlt = df[VLT_FEA].values
        self.df_sf = df[SF_FEA].values
        self.df_cat = df[CAT_FEA_HOT].values
        self.df_oth = df[MORE_FEA].values
        self.df_is = df[IS_FEA].values
        self.df_label = lb.values
        self.df_label_vlt = df[LABEL_vlt].values
        self.df_label_sf = df[LABEL_sf].values

        self.s2s_Enc_X = s2s['Enc_X'].values
        self.s2s_Enc_y = s2s['Enc_y'].values
        self.s2s_Dec_X = s2s['Dec_X'].values
        self.s2s_Dec_y = s2s['Dec_y'].values

        print(self.train_len, self.test_len)
                    

    def __getitem__(self, idx):

        # if self.phase=='test':
        #     index = self.test_indices[idx]
        # else:
        #     index = self.train_indices[idx]
        index = idx

        # Need optimize
        if self.device != 'cpu':
            a = torch.from_numpy(self.df_vlt[index]).float().to(self.device)
            b = torch.from_numpy(self.df_sf[index]).float().to(self.device)
            c = torch.from_numpy(self.df_cat[index]).float().to(self.device)
            d = torch.from_numpy(self.df_oth[index]).float().to(self.device)
            e = torch.from_numpy(self.df_is[index]).float().to(self.device)
            f = torch.from_numpy(self.df_label[index]).float().to(self.device)
            g = torch.from_numpy(self.df_label_vlt[index]).float().to(self.device)
            i = torch.FloatTensor(self.s2s_Enc_X[index]).view(rnn_hist_long,rnn_input_dim).to(self.device)
            j = torch.FloatTensor(self.s2s_Enc_y[index]).view(rnn_hist_long).to(self.device)
            k = torch.FloatTensor(self.s2s_Dec_X[index]).view(rnn_pred_long,rnn_input_dim).to(self.device)
            l = torch.FloatTensor(self.s2s_Dec_y[index]).view(rnn_pred_long).to(self.device)
        else:
            a = torch.from_numpy(self.df_vlt[index]).float()
            b = torch.from_numpy(self.df_sf[index]).float()
            c = torch.from_numpy(self.df_cat[index]).float()
            d = torch.from_numpy(self.df_oth[index]).float()
            e = torch.from_numpy(self.df_is[index]).float()
            f = torch.from_numpy(self.df_label[index]).float()
            g = torch.from_numpy(self.df_label_vlt[index]).float()
            i = torch.FloatTensor(self.s2s_Enc_X[index]).view(rnn_hist_long,rnn_input_dim)
            j = torch.FloatTensor(self.s2s_Enc_y[index]).view(rnn_hist_long)
            k = torch.FloatTensor(self.s2s_Dec_X[index]).view(rnn_pred_long,rnn_input_dim)
            l = torch.FloatTensor(self.s2s_Dec_y[index]).view(rnn_pred_long)
        return (a,b,c,d,e,f,g,i,j,k,l)
        
    def __len__(self):

        if self.phase=='train':
            return self.train_len
        else:
            return self.test_len



def get_loader(batch_size, device, eval=0, data_dir='../data/', num_workers=0):

    with open(data_dir + '1320_feature/df_e2e_tc.pkl', 'rb') as fp:
        o2 = pickle.load(fp)

    sku_set = o2.sku_id.unique()
    sku_train, sku_test = train_test_split(sku_set, random_state=10, train_size=0.8, test_size=0.2)

    # dset = E2E_Dataset(phase, device)
    if eval == 0:
        train_set = E2E_v6_Dataset(o2, sku_train, sku_test, 'train', device)
        data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              # pin_memory=True
                                              )
    else: data_loader = {}

    test_set = E2E_v6_Dataset(o2, sku_train, sku_test, 'test', device)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=test_set.test_len,
                                              shuffle=False,
                                              num_workers=0,
                                              # pin_memory=True
                                              )

    return data_loader, test_loader
  