# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-02 12:52:36


import tensorflow as tf
from model import End2End_v1
from train import Solver


flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS

def main(_):
    model = End2End_v1(mode=FLAGS.mode, learning_rate=0.0001)
    solver = Solver(model, batch_size=64, pretrain_iter=20000, train_epoch=30, eval_set=100, 
                    data_dir='../data/')
    
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
        pred = solver.eval()

if __name__ == '__main__':
    tf.app.run()