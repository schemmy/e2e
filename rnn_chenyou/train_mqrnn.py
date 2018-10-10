import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
# from PIL import Image
# from torch.nn.utils.rnn import pack_padded_sequence
import json
from data_loader import get_loader_price
from model_seq import MQ_RNN
from quantile_loss import QauntileLoss, QauntileLossNew
#from misc import AverageMeter
#from tensorboardX import SummaryWriter

# https://en.wikipedia.org/wiki/Linear_interpolation
def linear_interpolation(x0,y00,x1,y11,xs,use_sqrt=False):
    ys = []
    
    if use_sqrt:
        y0,y1 = torch.pow(y00,2),torch.pow(y11,2)
    else:
        y0,y1 = y00,y11
    
    rt = (y1-y0) / (x1-x0)
    for x in xs:
        y = y0 + (x-x0) * rt
        if use_sqrt:
            y = torch.sqrt(y)
        ys.append(y)
        
    return ys
    
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    ################
    # set parameters
    ################
    args.hidden_size = 30   # see paper, LSTM size
    args.context_size = 5   # see paper, 3.2, context size
    args.pred_long = 31     # see paper, the forecast is future 24 hours
    args.hist_long = 51    # see paper, the history is up to 168 hours
    args.total_long = args.pred_long + args.hist_long
    args.input_dim = 2    # two horizons

    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]    
    num_quantiles = len(quantiles)
    
    with open('../data/1320_feature/df_s2s.pkl', 'rb') as fp:
        data_json = pickle.load(fp)

    # Build data loader
    data_loader = get_loader_price('train', data_json, args.input_dim, args.pred_long, args.hist_long, args.total_long,
                             args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_loader = get_loader_price('test', data_json, args.input_dim, args.pred_long, args.hist_long, args.total_long,
                             1, shuffle=False, num_workers=1)
    
    #writer = SummaryWriter()

                                 
    rnn = MQ_RNN(args.input_dim, args.hidden_size, args.context_size, num_quantiles, args.pred_long, args.hist_long)
#     rnn = rnn.cuda()

    # Loss 
    quantile_loss = QauntileLossNew(y_norm=False, size_average=True, use_square=False).cuda()
    
    # Optimizer
    params = list(rnn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)

    if args.test==1:
        rnn.load_state_dict(torch.load('./models_price_convrnn/qr_16_l2.236.pkl'))



    for epoch in range(args.num_epochs):
        
        if args.test==0:      
            for i, (input_hist, target_hist, input_pred, target_pred) in enumerate(data_loader):
                # print(input_hist.shape, target_hist.shape, input_pred.shape, target_pred.shape)
   
#                 input_hist = input_hist.cuda()
#                 target_hist = target_hist.cuda()
#                 input_pred = input_pred.cuda()
#                 target_pred = target_pred.cuda()
            
                # Forward, Backward and Optimize
                outputs, targets = rnn(input_hist, target_hist, input_pred, target_pred)
                loss = torch.tensor(0.0)#.cuda()
            
                for k in range(num_quantiles):
                    loss += quantile_loss(outputs[:,:,k], targets, quantiles[k])
                    #loss_inter += quantile_loss(outputs_inter[:,:,k], targets_inter,  quantiles[k])
                loss /= num_quantiles
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if i%100==0:
                    print('Epoch %d Iter %d/%d, loss %.3f' % (epoch,i,len(data_loader),loss.item()))
            
            
#         rnn.eval()
#         with torch.no_grad():
            
#             loss = torch.tensor(0.0)#.cuda()
#             #loss_each = [torch.tensor(0.0).cuda() for k in range(num_quantiles)]
#             loss_each = [torch.tensor(0.0) for k in range(num_quantiles)]
            
#             for it, (input_hist, target_hist, input_pred, target_pred) in enumerate(test_loader):
            
# #                 input_hist = input_hist.cuda()
# #                 target_hist = target_hist.cuda()
# #                 input_pred = input_pred.cuda()
# #                 target_pred = target_pred.cuda()
    
#                 # Forward, Backward and Optimize
#                 outputs, targets = rnn(input_hist, target_hist, input_pred, target_pred)
                
#                 cnt = 0
#                 for k in range(num_quantiles-1):
#                     xs = []
#                     t0=quantiles[k]
#                     t0+=0.01
#                     while t0<quantiles[k+1]:
#                         t0 = round(t0,2)
#                         xs.append(t0)
#                         t0+=0.01
                    
#                     #print xs      
#                     ys = linear_interpolation(quantiles[k],outputs[:,:,k],quantiles[k+1],outputs[:,:,k+1],xs,use_sqrt=False)
#                     #print(ys[0].size())
#                     for x,y in zip(xs,ys):
#                         loss += quantile_loss(y, targets, x)
#                         cnt+=1
                
                
#                 for k in range(num_quantiles):
#                     t = quantile_loss(outputs[:,:,k], targets, quantiles[k])
#                     loss_each[k] += t
#                     loss += t
#                     cnt +=1
                
#                 #print(cnt)
#                 #assert cnt==99
            
            
            
#             loss_each_val = []
#             for k in range(num_quantiles):
#                 t = loss_each[k] / len(test_loader)
#                 loss_each_val.append(t.item())
                
#             loss_val = loss.item() / cnt / len(test_loader)
        
#             print('[Test] Epoch %d test loss %.3f' % (epoch, loss_val))
#             torch.save(rnn.state_dict(), os.path.join(args.model_path, 'qr_%d_l%.3f.pkl' %(epoch+1,loss_val)))

        rnn.train()   
           
        if args.test==1:
            exit()


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_price/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_path', type=str, default='./price_data.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    main(args)