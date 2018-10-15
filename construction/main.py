# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-15 15:36:06


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

    # model = End2End_v5_tc(device).to(device)
    model = End2End_v6_tc(device).to(device)

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

        data_loader, test_loader = get_loader(args.bs, device)
        
        # e2e_loss = E2E_loss()
        e2e_loss = E2E_v6_loss(device)
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)

        curr_epoch = 0

        # model.load_state_dict(torch.load('../logs/torch/e2e_v6_30.pkl'))
        # checkpoint = torch.load('../logs/torch/e2e_v6_30.pkl')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # curr_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # print('load model!')

        for epoch in range(curr_epoch, args.num_epochs):
            train_loss0, train_loss1 = 0, 0
            # for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, tar_sf) in enumerate(data_loader):
            for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, enc_X, enc_y, dec_X, dec_y) in enumerate(data_loader):

                # out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)
                out, out_vlt, out_sf = model(enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is)
                # batch_loss = e2e_loss(out_vlt, tar_vlt, out_sf, tar_sf, out, target)
                batch_loss1, batch_loss0 = e2e_loss(out_vlt, tar_vlt, out_sf, dec_y, out, target)

                # if epoch>0:
                optimizer.zero_grad()
                batch_loss0.backward()
                optimizer.step()

                train_loss1 += batch_loss1.item()
                train_loss0 += batch_loss0.item()

                if (i+1) % args.log_step == 0:

                    test_loss0, test_loss1 = 0, 0
                    # for _, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, tar_sf) in enumerate(test_loader):
                    for _, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, enc_X, enc_y, dec_X, dec_y) in enumerate(test_loader):
                        # out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)
                        out, out_vlt, out_sf = model(enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is)
                        # loss = e2e_loss(out_vlt, tar_vlt, out_sf, tar_sf, out, target)
                        loss1, loss0 = e2e_loss(out_vlt, tar_vlt, out_sf, dec_y, out, target)
                        test_loss1 += loss1.item()
                        test_loss0 += loss0.item()

                    print('Epoch %d %.3f%, loss_1 %.5f, loss_ttl %.5f, test_loss_1 %.5f, test_loss_ttl %.5f' % 
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

            
            if (epoch+1) % 10 == 0:
                # torch.save(model.state_dict(), os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))

        train_writer.close()
        test_writer.close()

    else:

        _, test_loader = get_loader(args.bs, device, eval=1)

        # model.load_state_dict(torch.load('../logs/torch/e2e_v6_20.pkl'))
        checkpoint = torch.load('../logs/torch/e2e_v6_30.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')

        for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, enc_X, enc_y, dec_X, dec_y) in enumerate(test_loader):
            # out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)
            out, out_vlt, out_sf = model(enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is)
        
        out = out.detach().cpu().numpy() / pd_scaler.loc[1, 'demand_RV'] + pd_scaler.loc[0, 'demand_RV']
        pred = pd.DataFrame(out, columns=['prediction']).fillna(0)
        pred.to_csv('../logs/torch/pred.csv', index=False)


if __name__ == '__main__':
    # tf.app.run()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_price/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_path', type=str, default='./price_data.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=198,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    main_tc(args)