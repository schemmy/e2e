# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-09 16:49:05


import tensorflow as tf
from model import *
from train import *
from loss import *
import argparse

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


    data_loader = get_loader('train', args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', 1, shuffle=False, num_workers=1)

    model = End2End_v5_tc()
    e2e_loss = E2E_loss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)

    if args.test==1:
        model.load_state_dict(torch.load('../logs/xx.pkl'))

    if args.test==0:      
        for epoch in range(args.num_epochs):

            train_loss = 0
            for i, (x_vlt, x_sf, x_cat, x_oth, x_is, target, tar_vlt, tar_sf) in enumerate(data_loader):

                out, out_vlt, out_sf = model(x_vlt, x_sf, x_cat, x_oth, x_is)

                loss = e2e_loss(out_vlt, tar_vlt, out_sf, tar_sf, out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            print('Epoch %d, loss %.3f' % (epoch, loss.item()))


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
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    main_tc(args)