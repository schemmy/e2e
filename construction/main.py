# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-16 18:07:09


import tensorflow as tf
from model import *
from train import *
from loss import *
import argparse
import time
from torch.multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)
from tensorboardX import SummaryWriter
from subprocess import call

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS


def main(_):

    out_dir = '../logs/'
    model = End2End_v5(mode=FLAGS.mode, learning_rate=0.0001)
    out_dir = out_dir + model.name + '/'
    test_path = out_dir+'checkpoint/'+model.name+'-20'
    solver = Solver(model, batch_size=64, pretrain_iter=20000, train_epoch=20, eval_set=100, 
                    data_dir='../data/', 
                    log_dir=out_dir,
                    model_save_path=out_dir,
                    test_model=test_path
                    )
    
    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)
    
    if FLAGS.mode == 'train':
        solver.train()
    elif FLAGS.mode == 'pretrain': 
        solver.pretrain()
    else:
        solver.eval()



def main_tc(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model_name == 'v5':
        model = End2End_v5_tc(device).to(device)
    elif args.model_name == 'v6':
        model = End2End_v6_tc(device).to(device)
    else:
        raise Exception('Unsupported model name!')

    if os.path.exists('../logs/torch_board/%s' %model.name):
        call(['rm', '-r', '../logs/torch_board/%s' %model.name])
    call(['mkdir', '../logs/torch_board/%s' %model.name])

    train_writer = SummaryWriter('../logs/torch_board/%s/train/' %model.name)
    test_writer = SummaryWriter('../logs/torch_board/%s/test/' %model.name)

    # enc_X = torch.Tensor(128,rnn_hist_long)
    # enc_y = torch.Tensor(128,rnn_hist_long)
    # dec_X = torch.Tensor(128,31,num_quantiles)
    # x_vlt = torch.Tensor(128,len(VLT_FEA))
    # x_cat = torch.Tensor(128,len(CAT_FEA_HOT))
    # x_oth = torch.Tensor(128,len(MORE_FEA))
    # x_is = torch.Tensor(128,len(IS_FEA))
    # rubish = torch.Tensor(128,30)
    # writer.add_graph(model, (enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is), verbose=False)


    # Train the Models
    # total_step = len(data_loader)

    if args.test==0:

        data_loader, test_loader = get_loader(args.bs, device, model.name)
        
        if 'v5' in model.name:
            e2e_loss = E2E_loss(device)
        elif 'v6' in model.name:
            e2e_loss = E2E_v6_loss(device)

        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)

        curr_epoch = 0

        if args.train_check != 'None':
            # model.load_state_dict(torch.load('../logs/torch/e2e_v6_30.pkl'))
            checkpoint = torch.load(args.model_path+args.train_check)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('load model!')

        for epoch in range(curr_epoch, args.num_epochs):
            train_loss0, train_loss1 = 0, 0
            # for i, X in enumerate(data_loader):
            for i, (X, S1, S2) in enumerate(data_loader):
                # out, out_vlt, out_sf = model(X[:,:50], X[:,50:112], X[:,112:124], X[:,124:125], X[:,125:129])
                out, out_vlt, out_sf = model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:50], X[:,112:124], X[:,124:125], X[:,125:129])
                # batch_loss1, batch_loss0 = e2e_loss(out_vlt, X[:,130:131], out_sf, X[:,131:132], out, X[:,129:130])
                batch_loss1, batch_loss0 = e2e_loss(out_vlt, X[:,130:131], out_sf, S2[:,:,2], out, X[:,129:130])

                # if epoch>0:
                optimizer.zero_grad()
                batch_loss0.backward()
                optimizer.step()

                train_loss1 += batch_loss1.item()
                train_loss0 += batch_loss0.item()

                if (i+1) % args.log_step == 0:

                    test_loss0, test_loss1 = 0, 0
                    # for _, X in enumerate(test_loader):
                    for _, (X, S1, S2) in enumerate(test_loader):
                        # out, out_vlt, out_sf = model(X[:,:50], X[:,50:112], X[:,112:124], X[:,124:125], X[:,125:129])
                        out, out_vlt, out_sf = model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:50], X[:,112:124], X[:,124:125], X[:,125:129])
                        # loss1, loss0 = e2e_loss(out_vlt, X[:,130:131], out_sf, X[:,131:132], out, X[:,129:130])
                        loss1, loss0 = e2e_loss(out_vlt, X[:,130:131], out_sf, S2[:,:,2], out, X[:,129:130])
                        test_loss1 += loss1.item()
                        test_loss0 += loss0.item()

                    print('Epoch %d pct %.3f, loss_1 %.5f, loss_ttl %.5f, test_loss_1 %.5f, test_loss_ttl %.5f' % 
                                    (epoch,(i+1)/len(data_loader),train_loss1/args.log_step,train_loss0/args.log_step,
                                        test_loss1/len(test_loader), test_loss0/len(test_loader)))

                    for name, param in model.named_parameters():
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_out',train_loss1/args.log_step, epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_ttl',train_loss0/args.log_step, epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_out', test_loss1/len(test_loader), epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_ttl', test_loss0/len(test_loader), epoch*len(data_loader)+i)
                    train_writer.export_scalars_to_json('../logs/torch_board/%s/train/scalars_train.json' %model.name)
                    test_writer.export_scalars_to_json('../logs/torch_board/%s/test/scalars_test.json' %model.name)

                    train_loss0, train_loss1 = 0, 0

            
            if (epoch+1) % args.save_step == 0:
                # torch.save(model.state_dict(), os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': batch_loss1,
                            }, os.path.join('../logs/torch/', 'e2e_%s_%d.pkl' %(model.name,epoch+1)))

        train_writer.close()
        test_writer.close()

    else:

        _, test_loader = get_loader(args.bs, device, model.name, eval=1)

        # model.load_state_dict(torch.load('../logs/torch/e2e_v6_20.pkl'))
        checkpoint = torch.load(args.model_path+args.model_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        
        # for _, X in enumerate(test_loader):
        for _, (X, S1, S2) in enumerate(test_loader):
            # out, out_vlt, out_sf = model(X[:,:50], X[:,50:112], X[:,112:124], X[:,124:125], X[:,125:129])
            out, out_vlt, out_sf = model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:50], X[:,112:124], X[:,124:125], X[:,125:129])

        pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')
        out = out.detach().cpu().numpy() / pd_scaler.loc[1, LABEL[0]] + pd_scaler.loc[0, LABEL[0]]
        pred = pd.DataFrame(out, columns=['E2E_NN_pred'])
        if 'v5' in model.name:
            out_sf = out_sf.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_sf[0]] + pd_scaler.loc[0, LABEL_sf[0]]
            pred_sf = pd.DataFrame(out_sf, columns=['E2E_NN_SF_mean_pred'])
            pred = pd.concat([pred, pred_sf], axis=1)
            pred.to_csv('../logs/torch/pred.csv', index=False)
        else:
            out_sf = out_sf.view(-1, rnn_pred_long, num_quantiles).detach().cpu().numpy()
            out_sf = np.exp(out_sf) - 1
            pred.to_csv('../logs/torch/pred.csv', index=False)
            out_sf.dump('../logs/torch/pred_E2E_NN_RNN.pkl')



if __name__ == '__main__':
    # tf.app.run()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str , default='v6',
                        help='v5: MLP; v6: RNN')
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--model_to_load', type=str , default='e2e_v6_tc_20.pkl',
                        help='model to be loaded for evaluation')
    parser.add_argument('--train_check', type=str , default='None',
                        help='checkpoint for continuing training')
    parser.add_argument('--model_path', type=str, default='../logs/torch/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_path', type=str, default='../data/1320_feature/',
                        help='path for data')
    parser.add_argument('--log_step', type=int , default=145,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=10,
                        help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    main_tc(args)