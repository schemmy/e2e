# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-19 15:45:09

import sys
sys.path.append('../')
# import tensorflow as tf
from trainers.train_tc import *
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.model_name == 'v5':
        model = End2End_v5_tc(device).to(device)
    elif args.model_name == 'v6':
        model = End2End_v6_tc(device).to(device)
    elif args.model_name == 'v7':
        model = End2End_v7_tc(device).to(device)
    else:
        raise Exception('Unsupported model name!')

    trainer = Trainer(model, args, device)

    if args.model_name == 'v5':
        if args.test == 0:
            trainer.train_v5_tc()
        else:
            trainer.eval_v5_tc()
    else:
        if args.test == 0:
            trainer.train_v6_tc()
        else:
            trainer.eval_v6_tc()



if __name__ == '__main__':
    # tf.app.run()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str , default='v5',
                        help='v5: MLP; v6: RNN')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--test_sku', type=str, default='None')
    parser.add_argument('--model_to_load', type=str , default='e2e_v6_tc_40.pkl',
                        help='model to be loaded for evaluation')
    parser.add_argument('--train_check', type=str , default='None',
                        help='checkpoint for continuing training')
    parser.add_argument('--model_path', type=str, default='../logs/torch/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_path', type=str, default='../data/1320_feature/',
                        help='path for data')
    parser.add_argument('--log_path', type=str, default='../logs/torch_board/',
                        help='path for data')
    parser.add_argument('--log_step', type=int , default=145,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=5,
                        help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    main_tc(args)