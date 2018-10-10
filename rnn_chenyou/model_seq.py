import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F


class Decoder_MLP(nn.Module):

    def __init__(self, x_dim, hidden_size, context_size, num_quantiles, pred_long):
        """Set the hyper-parameters and build the layers."""
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
        """Initialize weights."""
        self.global_mlp.weight.data.normal_(0.0, 0.02)
        self.global_mlp.bias.data.fill_(0)

        self.local_mlp.weight.data.normal_(0.0, 0.02)
        self.local_mlp.bias.data.fill_(0)

        
        
    def forward(self, hidden_states, x_future):
        # hidden_states: N x hidden_size
        # x_future:      N x pred_long x x_dim
        # y_future:      N x pred_long

        #print('x_future size 0 ',x_future.size())
        x_future_1 = x_future.view(x_future.size(0),x_future.size(1)*x_future.size(2))
        #print('x_future size 1 ',x_future.size())
        #print('hidden_states size 0',hidden_states.size())

        # global MLP
        hidden_future_concat = torch.cat((hidden_states, x_future_1),dim=1)
        context_vectors = F.sigmoid( self.global_mlp(hidden_future_concat) )
        #print('context_vectors size ',context_vectors.size())

        ca = context_vectors[:, self.context_size*self.pred_long:]
        #print('ca size ',ca.size())
        
        results = []        
        for k in range(self.pred_long):
            xk = x_future[:,k,:]
            ck = context_vectors[:, k*self.context_size:(k+1)*self.context_size]
            cak = torch.cat((ck,ca,xk),dim=1)
            #print('cak size ', cak.size())
            # local MLP
            quantile_pred = self.local_mlp(cak)
            quantile_pred = quantile_pred.view(quantile_pred.size(0),1,quantile_pred.size(1))
            #print('quantile_pred size ',quantile_pred.size())
            results.append(quantile_pred)
            
            
        result = torch.cat(results,dim=1)
        #print('result size ',result.size())  # should be N x pred_long x num_quantiles
        
        return result
    
    

class MQ_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, context_size, num_quantiles, pred_long, hist_long, num_layers=1):
        """Set the hyper-parameters and build the layers."""
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
        """Initialize weights."""
        self.decoder.init_weights()
        
        self.linear_encoder.weight.data.normal_(0.0, 0.02)
        self.linear_encoder.bias.data.fill_(0)

    
    
    # y_seq_hist size: N x hist_long x 2
    # y_seq_hist size: N x hist_long
    # y_seq_pred size: N x pred_long x 2
    # y_seq_pred size: N x pred_long
    
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
                
        #print('xy seq hist size ', xy_seq_hist.size())

        hiddens, (ht, c) = self.lstm(x_total_hist)
        #print('LSTM total output hidden size ', hiddens.size())
        #print('LSTM last output hidden size ', ht.size())
        
        ht = ht.view(ht.size(1),ht.size(2))

        result = self.decoder(ht, x_total_pred)
        
        #print('MQ_RNN pred result size', result.size())
        #print('y_future_quantile size', y_future_quantile.size())

        return result, y_seq_pred


class CONV_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, context_size, num_quantiles, pred_long, hist_long, num_layers=2):
        """Set the hyper-parameters and build the layers."""
        super(CONV_RNN, self).__init__()

        #self.decoder = Decoder_MLP(input_dim, hidden_size, context_size, num_quantiles, pred_long)
        self.lstm = nn.LSTM(context_size+1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.lstm_decoder = nn.LSTM(context_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)

        self.linear_encoder = nn.Linear(input_dim, context_size)  

        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        self.num_layers = num_layers
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = pred_long + hist_long
        
        self.inter_classifier = nn.Linear(hidden_size, num_quantiles)
        self.final_classifier = nn.Linear(hidden_size*2, num_quantiles)
        self.conv2d = nn.Conv2d(1, 1, [5,1], stride=1, padding=[2,0])
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        #self.decoder.init_weights()
        
        self.conv2d.weight.data.normal_(0.0, 0.02)
        self.conv2d.bias.data.fill_(0)

        self.linear_encoder.weight.data.normal_(0.0, 0.02)
        self.linear_encoder.bias.data.fill_(0)
        
        
        self.inter_classifier.weight.data.normal_(0.0, 0.02)
        self.inter_classifier.bias.data.fill_(0)

        self.final_classifier.weight.data.normal_(0.0, 0.02)
        self.final_classifier.bias.data.fill_(0)
        
        
    
    # y_seq_hist size: N x hist_long x 2
    # y_seq_hist size: N x hist_long
    # y_seq_pred size: N x pred_long x 2
    # y_seq_pred size: N x pred_long
    
    def forward(self, x_seq_hist, y_seq_hist, x_seq_pred, y_seq_pred):
        
        bsize = x_seq_hist.size(0)
        #print(x_seq_hist.size(), x_seq_pred.size())
        assert x_seq_hist.size(1) == self.hist_long
        assert x_seq_hist.size(2) == self.input_dim
        assert x_seq_pred.size(1) == self.pred_long
        assert y_seq_pred.size(1) == self.pred_long


        x_feat_hist = F.tanh(self.linear_encoder(x_seq_hist))
        x_feat_pred = F.tanh(self.linear_encoder(x_seq_pred))
        
        
        y_seq_hist_1 = y_seq_hist.view(y_seq_hist.size(0), y_seq_hist.size(1), 1)
        

        x_total_hist =  torch.cat([x_feat_hist,y_seq_hist_1], dim=2)
        x_total_pred =  torch.cat([x_feat_pred], dim=2)
        
        
        self.lstm.flatten_parameters()
        
        hiddens, (ht, c) = self.lstm(x_total_hist)
        #print('LSTM total output hidden size ', hiddens.size())  # (32, 31, 30)
        #print('LSTM last output hidden size ', ht.size())       # (2, 32, 30)
        
        if self.num_layers == 1:
            ht_out = ht.view(ht.size(1),ht.size(2))
        else:
            ht_out = torch.cat( (ht[0,:,:], ht[1,:,:]), dim=1 )
            
        
        ht_decoder = torch.zeros(self.num_layers * 2, bsize, self.hidden_size).cuda()
        ct_decoder = torch.zeros(self.num_layers * 2, bsize, self.hidden_size).cuda()
        
        ht_decoder[:2,:,:] = ht
        ct_decoder[:2,:,:] = c
        
        hiddens_out, _ = self.lstm_decoder( x_total_pred, (ht_decoder, ct_decoder) )
        #print('hiddens_out size', hiddens_out.size())
        hiddens_out = hiddens_out.view(hiddens_out.size(0),1,hiddens_out.size(1),hiddens_out.size(2))
        #print('hiddens_out size', hiddens_out.size())
        hiddens_out_conv = self.conv2d(hiddens_out)
        #print('hiddens_out_conv size', hiddens_out_conv.size())
        hiddens_out_conv_2 = hiddens_out.view(hiddens_out.size(0),hiddens_out.size(2),hiddens_out.size(3))
        #print('hiddens_out_conv_2 size', hiddens_out_conv_2.size())

        result = self.final_classifier(hiddens_out_conv_2)
        
        
        y_seq_hist_pred = self.inter_classifier(hiddens[:,:-1,:])
        y_seq_hist_gt = y_seq_hist.detach() 
        y_seq_hist_gt = y_seq_hist_gt[:,1:]  # pred next
        
        #print('Size 1,2',y_seq_hist_pred.size(),y_seq_hist_gt.size())
        
        #print('MQ_RNN pred result size', result.size())         # (32, 31)
        #print('Y size', y_seq_pred.size())                      # (32, 31)

        return result, y_seq_pred, y_seq_hist_pred, y_seq_hist_gt   


class CONV_RNN_MULTI(nn.Module):

    def __init__(self, input_dim, hidden_size, context_size, num_quantiles, pred_long, hist_long, num_layers=2):
        """Set the hyper-parameters and build the layers."""
        super(CONV_RNN_MULTI, self).__init__()

        #self.decoder = Decoder_MLP(input_dim, hidden_size, context_size, num_quantiles, pred_long)
        self.lstm = nn.LSTM(context_size+1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.lstm_decoder = nn.LSTM(context_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)

        self.linear_encoder = nn.Linear(input_dim, context_size)  

        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        self.num_layers = num_layers
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = pred_long + hist_long
        
        self.inter_classifier = nn.Linear(hidden_size, num_quantiles)
        self.final_classifier = nn.Linear(hidden_size*2*3, num_quantiles)
        self.conv2d_1 = nn.Conv2d(1, 1, [3,1], stride=1, padding=[1,0])
        self.conv2d_2 = nn.Conv2d(1, 1, [5,1], stride=1, padding=[2,0])
        self.conv2d_3 = nn.Conv2d(1, 1, [7,1], stride=1, padding=[3,0])

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        #self.decoder.init_weights()
        
        self.conv2d_1.weight.data.normal_(0.0, 0.02)
        self.conv2d_1.bias.data.fill_(0)

        self.conv2d_2.weight.data.normal_(0.0, 0.02)
        self.conv2d_2.bias.data.fill_(0)
        
        self.conv2d_3.weight.data.normal_(0.0, 0.02)
        self.conv2d_3.bias.data.fill_(0)
        
        
        self.linear_encoder.weight.data.normal_(0.0, 0.02)
        self.linear_encoder.bias.data.fill_(0)
        
        
        self.inter_classifier.weight.data.normal_(0.0, 0.02)
        self.inter_classifier.bias.data.fill_(0)

        self.final_classifier.weight.data.normal_(0.0, 0.02)
        self.final_classifier.bias.data.fill_(0)
        
        
    
    # y_seq_hist size: N x hist_long x 2
    # y_seq_hist size: N x hist_long
    # y_seq_pred size: N x pred_long x 2
    # y_seq_pred size: N x pred_long
    
    def forward(self, x_seq_hist, y_seq_hist, x_seq_pred, y_seq_pred):
        
        bsize = x_seq_hist.size(0)
        #print(x_seq_hist.size(), x_seq_pred.size())
        assert x_seq_hist.size(1) == self.hist_long
        assert x_seq_hist.size(2) == self.input_dim
        assert x_seq_pred.size(1) == self.pred_long
        assert y_seq_pred.size(1) == self.pred_long


        x_feat_hist = F.tanh(self.linear_encoder(x_seq_hist))
        x_feat_pred = F.tanh(self.linear_encoder(x_seq_pred))
        
        
        y_seq_hist_1 = y_seq_hist.view(y_seq_hist.size(0), y_seq_hist.size(1), 1)
        

        x_total_hist =  torch.cat([x_feat_hist,y_seq_hist_1], dim=2)
        x_total_pred =  torch.cat([x_feat_pred], dim=2)
        
        
        self.lstm.flatten_parameters()
        
        hiddens, (ht, c) = self.lstm(x_total_hist)
        #print('LSTM total output hidden size ', hiddens.size())  # (32, 31, 30)
        #print('LSTM last output hidden size ', ht.size())       # (2, 32, 30)
        
        if self.num_layers == 1:
            ht_out = ht.view(ht.size(1),ht.size(2))
        else:
            ht_out = torch.cat( (ht[0,:,:], ht[1,:,:]), dim=1 )
            
        
        ht_decoder = torch.zeros(self.num_layers * 2, bsize, self.hidden_size).cuda()
        ct_decoder = torch.zeros(self.num_layers * 2, bsize, self.hidden_size).cuda()
        
        ht_decoder[:2,:,:] = ht
        ct_decoder[:2,:,:] = c
        
        hiddens_out, _ = self.lstm_decoder( x_total_pred, (ht_decoder, ct_decoder) )
        #print('hiddens_out size', hiddens_out.size())
        hiddens_out = hiddens_out.view(hiddens_out.size(0),1,hiddens_out.size(1),hiddens_out.size(2))
        #print('hiddens_out size', hiddens_out.size())
        hiddens_out_conv_1 = self.conv2d_1(hiddens_out)
        hiddens_out_conv_2 = self.conv2d_2(hiddens_out)
        hiddens_out_conv_3 = self.conv2d_3(hiddens_out)
        hiddens_out_all = torch.cat([hiddens_out_conv_1,hiddens_out_conv_2,hiddens_out_conv_3],dim=3)
        #print('hiddens_out_conv size', hiddens_out_conv.size())
        hiddens_out_conv_2 = hiddens_out_all.view(hiddens_out_all.size(0),hiddens_out_all.size(2),hiddens_out_all.size(3))
        #print('hiddens_out_conv_2 size', hiddens_out_conv_2.size())

        result = self.final_classifier(hiddens_out_conv_2)
        
        
        y_seq_hist_pred = self.inter_classifier(hiddens[:,:-1,:])
        y_seq_hist_gt = y_seq_hist.detach() 
        y_seq_hist_gt = y_seq_hist_gt[:,1:]  # pred next
        
        #print('Size 1,2',y_seq_hist_pred.size(),y_seq_hist_gt.size())
        
        #print('MQ_RNN pred result size', result.size())         # (32, 31)
        #print('Y size', y_seq_pred.size())                      # (32, 31)

        return result, y_seq_pred, y_seq_hist_pred, y_seq_hist_gt   
