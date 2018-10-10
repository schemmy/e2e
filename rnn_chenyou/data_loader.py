import torch
#import torchvision.transforms as transforms
import torch.utils.data as data
import os
#import pickle
import numpy as np
#from PIL import Image
import random
import json
from sklearn.model_selection import train_test_split

class GEFPriceDataset(data.Dataset):

    def __init__(self, phase, df, input_dim, pred_long, hist_long, total_long, test_sample=12):
        df['Dec_y'] = df['Dec_y'].apply(lambda x:  [x[0]+[x[0][-1]]*(31-len(x[0]))] 
                                                if len(x[0])<31 else x)
        self.data = df
        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = total_long
        self.input_dim = input_dim
        self.test_sample = test_sample
        self.phase = phase

        # get 12 different forecast creation dates
        #if phase=='test':
        sku_set = df.sku_id.unique()
        sku_train, sku_test = train_test_split(sku_set, random_state=10, train_size=0.8, test_size=0.2)

        self.train_indices = list(df[df['sku_id'].isin(sku_train)].index)
        self.test_indices = list(df[df['sku_id'].isin(sku_test)].index)

        self.train_len = len(self.train_indices)
        self.test_len = len(self.test_indices)

        print(self.train_len, self.test_len)
                    
    def __getitem__(self, idx):
        
        if self.phase=='test':
            index = self.test_indices[idx]
        else:
            index = self.train_indices[idx]

        Enc_input = torch.FloatTensor(self.data.loc[index, 'Enc_X']).view(self.hist_long,self.input_dim)
        Enc_target = torch.FloatTensor(self.data.loc[index, 'Enc_y']).view(self.hist_long)
        Dec_input = torch.FloatTensor(self.data.loc[index, 'Dec_X']).view(self.pred_long,self.input_dim)
        Dec_target = torch.FloatTensor(self.data.loc[index, 'Dec_y']).view(self.pred_long)

        return Enc_input, Enc_target, Dec_input, Dec_target
        
    def __len__(self):
        if self.phase=='train':
            return len(self.train_indices)
        else:
            return len(self.test_indices)




def get_loader_price(phase, data_json, input_dim, pred_long, hist_long, total_long, batch_size, shuffle, num_workers):

    dset = GEFPriceDataset(phase, data_json, input_dim, pred_long, hist_long, total_long)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=shuffle)
    return data_loader