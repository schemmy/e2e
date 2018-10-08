import torch
#import torchvision.transforms as transforms
import torch.utils.data as data
import os
#import pickle
import numpy as np
#from PIL import Image
import random
import json

class GEFPriceDataset(data.Dataset):

    def __init__(self, phase, data_json, input_dim, pred_long, hist_long, total_long, test_sample=12):
        self.price_data = data_json
        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = total_long
        self.input_dim = input_dim
        self.test_sample = test_sample
        self.phase = phase

        # get 12 different forecast creation dates
        #if phase=='test':
        ll = len(self.price_data) - total_long # hist_long is 7*24
        assert ll%24==0
        numL = ll/24
        print(numL)
        numl0 = (numL-7) // test_sample
        

        tmp1 = range(7,numL)
        tmp2 = []
        tmp3 = []
        for k in range(1,self.test_sample+1):
            tt = k*numl0
            for p in range(7):
                tmp2.append(tt+p)
                
                tmp3.append(tt+p)
                tmp3.append(tt-p)
                
        tmp0 = set(tmp1)-set(tmp3)
        
        #print(len(tmp0),len(tmp2))
        self.train_indices = [a*24+1 for a in tmp0]
        self.test_indices = [a*24+1 for a in tmp2]
        
        
        #print self.train_indices,'--train'
        #print
        #print self.test_indices,'--test'

        print(len(self.price_data),len(self.train_indices),len(self.test_indices))
                    
    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        
        if self.phase=='test':
            index = self.test_indices[idx] - self.hist_long
        else:
            index = self.train_indices[idx] - self.hist_long

        input = torch.zeros((1,self.total_long,self.input_dim), dtype=torch.float)
        target = torch.zeros((1,self.total_long), dtype=torch.float)

        for k in xrange(self.total_long):
            data = self.price_data[index+k]
            input[0,k,:] = torch.tensor(np.array(data[1:1+self.input_dim]))
            target[0,k] = data[self.input_dim+1]
        
        #print(target)
        return input[0,:self.hist_long,:], target[0,:self.hist_long], input[0,self.hist_long:,:], target[0,self.hist_long:]
        
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