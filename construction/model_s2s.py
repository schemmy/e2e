# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-10 11:26:23
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-10 11:50:40


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F



class Decoder_MLP(nn.Module):

    def __init__(self, x_dim, hidden_size, context_size, num_quantiles, pred_long):

        super(Decoder_MLP, self).__init__()
        
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_quantiles = num_quantiles
        self.pred_long = pred_long

        self.global_mlp = nn.Linear(hidden_size + x_dim * pred_long, context_size*(pred_long+1))
        self.local_mlp = nn.Linear(context_size * 2 + x_dim, num_quantiles)

        self.init_weights()
    
    def init_weights(self):

        self.global_mlp.weight.data.normal_(0.0, 0.02)
        self.global_mlp.bias.data.fill_(0)

        self.local_mlp.weight.data.normal_(0.0, 0.02)
        self.local_mlp.bias.data.fill_(0)
        
        
    def forward(self, hidden_states, x_future):
        # hidden_states: N x hidden_size
        # x_future:      N x pred_long x x_dim
        # y_future:      N x pred_long

        x_future_1 = x_future.view(x_future.size(0),x_future.size(1)*x_future.size(2))

        hidden_future_concat = torch.cat((hidden_states, x_future_1),dim=1)
        context_vectors = F.sigmoid( self.global_mlp(hidden_future_concat) )

        ca = context_vectors[:, self.context_size*self.pred_long:]
        
        results = []        
        for k in range(self.pred_long):
            xk = x_future[:,k,:]
            ck = context_vectors[:, k*self.context_size:(k+1)*self.context_size]
            cak = torch.cat((ck,ca,xk),dim=1)

            quantile_pred = self.local_mlp(cak)
            quantile_pred = quantile_pred.view(quantile_pred.size(0),1,quantile_pred.size(1))

            results.append(quantile_pred)
            
            
        result = torch.cat(results,dim=1)
        
        return result
    
    

class MQ_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, context_size, num_quantiles, pred_long, hist_long, num_layers=1):

        super(MQ_RNN, self).__init__()

        self.decoder = Decoder_MLP(context_size, hidden_size, context_size, num_quantiles, pred_long)
        self.lstm = nn.LSTM(context_size+1, hidden_size, num_layers, batch_first=True)
        
        self.linear_encoder = nn.Linear(input_dim, context_size)  

        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = pred_long + hist_long
        self.init_weights()
    
    def init_weights(self):

        self.decoder.init_weights()
        
        self.linear_encoder.weight.data.normal_(0.0, 0.02)
        self.linear_encoder.bias.data.fill_(0)


    
    def forward(self, x_seq_hist, y_seq_hist, x_seq_pred, y_seq_pred):
        
        bsize = x_seq_hist.size(0)
        #print(x_seq_hist.size(), x_seq_pred.size())
        assert x_seq_hist.size(1) == self.hist_long
        assert x_seq_hist.size(2) == self.input_dim
        assert x_seq_pred.size(1) == self.pred_long
        assert y_seq_pred.size(1) == self.pred_long

        x_feat_hist = F.tanh(self.linear_encoder(x_seq_hist))
        x_feat_pred = F.tanh(self.linear_encoder(x_seq_pred))
        
        
        self.lstm.flatten_parameters()
        
        y_seq_hist_1 = y_seq_hist.view(y_seq_hist.size(0), y_seq_hist.size(1), 1)

        x_total_hist =  torch.cat([x_feat_hist,y_seq_hist_1], dim=2)
        x_total_pred =  torch.cat([x_feat_pred], dim=2)
                

        hiddens, (ht, c) = self.lstm(x_total_hist)
        
        ht = ht.view(ht.size(1),ht.size(2))

        result = self.decoder(ht, x_total_pred)
        
        return result, y_seq_pred
