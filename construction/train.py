# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:04:49
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-03 11:09:44


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd
import pickle
import os
from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=20000, train_epoch=20, eval_set=100, 
                 data_dir='../data/', 
                 log_dir='../logs/',
                 model_save_path='../logs/', 
                 # pretrained_model='logs/model/svhn_model-20000', 
                 test_model='../logs/v1/checkpoint/v1-20'
                 ):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_epoch = train_epoch
        self.eval_set = eval_set
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        # self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True


    def read_data(self):

        with open(self.data_dir+'1320_feature/df_train_test.pkl', 'rb') as fp:
            df_train, df_test = pickle.load(fp)

        X_train_ns, y_train_ns, id_train = df_train[SCALE_FEA], df_train[LABEL], df_train[IDX]
        X_test_ns, y_test_ns, id_test = df_test[SCALE_FEA], df_test[LABEL], df_test[IDX]

        self.n_train, self.n_test = len(X_train_ns), len(X_test_ns)
        self.X_scaler = MinMaxScaler() # For normalizing dataset
        self.y_scaler = MinMaxScaler() # For normalizing dataset
        # We want to predict Close value of stock 
        self.X_train = pd.DataFrame(self.X_scaler.fit_transform(X_train_ns), columns=X_train_ns.columns)
        self.y_train = pd.DataFrame(self.y_scaler.fit_transform(y_train_ns), columns=y_train_ns.columns)

        self.X_test = pd.DataFrame(self.X_scaler.transform(X_test_ns), columns=X_test_ns.columns)
        self.y_test = pd.DataFrame(self.y_scaler.transform(y_test_ns), columns=y_test_ns.columns)
        # pd.DataFrame(min_max_scaler.inverse_transform(y_test), columns=y_test.columns)



    def train(self):

        # build a graph
        model = self.model
        model.build_model()

        print ('Reading data..!')
        self.read_data()

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()

            # print ('loading pretrained model F..')
            # variables_to_restore = slim.get_model_variables(scope='content_extractor')
            # restorer = tf.train.Saver(variables_to_restore)
            # restorer.restore(sess, self.pretrained_model)
            summary_writer_train = tf.summary.FileWriter(logdir=self.log_dir+'/train', graph=tf.get_default_graph())
            summary_writer_test = tf.summary.FileWriter(logdir=self.log_dir+'/test', graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')
            train_err, train_loss_summary = sess.run([model.loss, model.summary_op_trg],
                                      feed_dict={model.x_vlt: self.X_train[VLT_FEA].values, 
                                                 model.x_sf: self.X_train[SF_FEA].values, 
                                                 model.x_oth: self.X_train[MORE_FEA].values, 
                                                 model.x_cat: self.X_train[CAT_FEA_HOT].values, 
                                                 model.x_is: self.X_train[IS_FEA].values, 
                                                 model.y: self.y_train.values})
            test_err, test_loss_summary = sess.run([model.loss, model.summary_op_trg], 
                                                feed_dict={model.x_vlt: self.X_test[VLT_FEA].values, 
                                                 model.x_sf: self.X_test[SF_FEA].values, 
                                                 model.x_oth: self.X_test[MORE_FEA].values, 
                                                 model.x_cat: self.X_test[CAT_FEA_HOT].values, 
                                                 model.x_is: self.X_test[IS_FEA].values, 
                                                 model.y: self.y_test.values})
            summary_writer_train.add_summary(train_loss_summary, 0)
            summary_writer_test.add_summary(test_loss_summary, 0)
            print(0, train_err, test_err)
            it = 0

            for epoch in range(1, self.train_epoch+1):
                
                train_err = 0
                idx_ub = int(np.ceil(self.n_train/self.batch_size))
                for _ in range(0, idx_ub):
                    idx = np.random.randint(idx_ub)*self.batch_size
                    bs = min(idx + self.batch_size, self.n_train) - idx
                    batch_data = self.X_train.iloc[idx:bs+idx, :]
                    batch_labels = self.y_train.iloc[idx:bs+idx, :]
                    feed_dict = {model.x_vlt: batch_data[VLT_FEA].values, 
                                 model.x_sf: batch_data[SF_FEA].values, 
                                 model.x_oth: batch_data[MORE_FEA].values,
                                 model.x_cat: batch_data[CAT_FEA_HOT].values,
                                 model.x_is: batch_data[IS_FEA].values,
                                 model.y: batch_labels}
                    _, c_loss, train_loss_summary = sess.run([model.train_step, model.loss, model.summary_op_trg], feed_dict)
                    train_err += c_loss*bs
                    it += 1
                    summary_writer_train.add_summary(train_loss_summary, it)

                test_err, test_loss_summary = sess.run([model.loss, model.summary_op_trg], 
                                          feed_dict={model.x_vlt: self.X_test[VLT_FEA].values, 
                                                     model.x_sf: self.X_test[SF_FEA].values, 
                                                     model.x_oth: self.X_test[MORE_FEA].values, 
                                                     model.x_cat: self.X_test[CAT_FEA_HOT].values, 
                                                     model.x_is: self.X_test[IS_FEA].values, 
                                                     model.y: self.y_test.values})

                train_err /= self.n_train
                summary_writer_test.add_summary(test_loss_summary, it)
                print(epoch, train_err, test_err)

                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(self.model_save_path+'checkpoint/', self.model.name), global_step=epoch)
                    print('model/N2N_%s_%d saved' %(self.model.name, epoch))  


    def eval(self):
        # build model
        model = self.model
        model.build_model()

        print('Reading data..!')
        self.read_data()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start prediction..!')
            pred = sess.run(model.output, feed_dict={model.x_vlt: self.X_test[VLT_FEA].values, 
                                              model.x_sf: self.X_test[SF_FEA].values, 
                                              model.x_oth: self.X_test[MORE_FEA].values, 
                                              model.x_cat: self.X_test[CAT_FEA_HOT].values, 
                                              model.x_is: self.X_test[IS_FEA].values, 
                                              })

            pred = pd.DataFrame(self.y_scaler.inverse_transform(pred), columns=['prediction']).fillna(0)
        print(pred)
        pred.to_csv(self.model_save_path+'pred.csv', index=False)


