# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-09 20:21:32


import tensorflow as tf
from model import *
from train import *
from loss import *
import argparse
import time
from torch.multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)


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

    device = 'cpu'
    model = End2End_v5_tc()

    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)


    data_loader = get_loader('train', args.bs, args.gpu, device, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.bs, args.gpu, device, shuffle=False, num_workers=1)

    e2e_loss = E2E_loss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)


    if args.test==0:      
        for epoch in range(args.num_epochs):
            train_loss, test_loss = 0, 0
            start_time = time.time()
            for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, tar_sf) in enumerate(data_loader):

                out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)
                loss = e2e_loss(out_vlt, tar_vlt, out_sf, tar_sf, out, target)
                if epoch>0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                if i % 100 == 0:
                    print('Epoch %d Iter %d/%d, loss %.3f' % (epoch,i,len(data_loader),loss.item()))

            run = time.time() - start_time
            for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, tar_sf) in enumerate(test_loader):
                out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)
                loss = e2e_loss(out_vlt, tar_vlt, out_sf, tar_sf, out, target)
                test_loss += loss.item()

            print ('Epoch %d, time %s, train loss %.5f, test loss %.5f' 
                    %(epoch, run, train_loss/len(data_loader), 0/len(test_loader) ))

    else:
        model.load_state_dict(torch.load('../logs/xx.pkl'))





if __name__ == '__main__':
    # tf.app.run()

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
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--gpu', default=False, type=bool, help='GPU training')
    args = parser.parse_args()
    main_tc(args)