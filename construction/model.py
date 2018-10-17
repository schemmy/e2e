# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 15:45:51
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-15 18:31:21
# Template:
# https://github.com/yunjey/domain-transfer-network/blob/master/model.py

import tensorflow as tf
from config import *
from model_s2s import *
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')


class End2End_v1(object):


    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.name='v1'

    def build_model(self):
        
        if self.mode == 'pretrain':
            pass

        elif self.mode == 'train' or  self.mode == 'eval':

            self.cat_dim = len(CAT_FEA_HOT)
            self.vlt_dim = len(VLT_FEA)
            self.sf_dim = len(SF_FEA)
            self.oth_dim = len(MORE_FEA)
            self.is_dim = len(IS_FEA)
            self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim

            self.hidden_dim = [20, 8]

            self.output_dim = 1
            self.q = 0.9

            with tf.name_scope('Data'):
                self.x_vlt = tf.placeholder(tf.float32, shape=[None, self.vlt_dim], name='Input_vlt')
                self.x_sf = tf.placeholder(tf.float32, shape=[None, self.sf_dim], name='Input_sf')
                self.x_cat = tf.placeholder(tf.float32, shape=[None, self.cat_dim], name='Input_pf')
                self.x_oth = tf.placeholder(tf.float32, shape=[None, self.oth_dim], name='Input_more')
                self.x_is = tf.placeholder(tf.float32, shape=[None, self.is_dim], name='Input_IS')
                self.mean_vlt = tf.expand_dims(self.x_vlt[:,-17],1, name='mean_vlt')
                self.review_p = tf.expand_dims(self.x_oth[:,0],1, name='review_p')

            with tf.name_scope('Label'):
                self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='Label')


            with tf.variable_scope('Layer_1'):
                self.W1_vlt = tf.Variable(tf.truncated_normal([self.vlt_dim+self.sf_dim+self.cat_dim+self.oth_dim, self.hidden_dim[0]], 
                                                                stddev=0.001), name='Weight_1')
                self.b1_vlt = tf.Variable(tf.zeros([self.hidden_dim[0]]), name='Bias_1')
                self.l1 = tf.add(tf.matmul(tf.concat([self.x_vlt, self.x_sf, self.x_cat, self.x_oth], axis=1), self.W1_vlt), self.b1_vlt)
                self.l1 = tf.nn.relu(self.l1)

            with tf.variable_scope('Layer_2'):
                self.W2 = tf.Variable(tf.truncated_normal([self.hidden_dim[0], self.output_dim], stddev=0.001), name='Weight_2')
                self.b2 = tf.Variable(tf.zeros([self.hidden_dim[1]]), name='Bias_2')
                self.l2 = tf.add(tf.matmul(tf.concat([self.l1], axis=1), self.W2), self.b2)
                self.l2 = tf.nn.relu(self.l2)

            with tf.variable_scope('Layer_3'):
                self.W3 = tf.Variable(tf.truncated_normal([self.hidden_dim[1], self.output_dim], stddev=0.001), name='Weight_3')
                self.b3 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_3')
                self.output = tf.add(tf.matmul(tf.concat([self.l2], axis=1), self.W3), self.b3)
                self.error = self.y - self.output

            with tf.variable_scope('Loss'):
            #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
                self.c_os = self.q * self.error
                self.c_hd = (self.q-1) * self.error
                self.loss = tf.reduce_mean(tf.maximum(self.c_os, self.c_hd) )
            #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
            #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
            #     loss = tf.reduce_mean(c_os+ c_hd )    

            with tf.name_scope('Optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
            # back_summary = tf.summary.scalar('back_loss', self.c_os)
            # hold_summary = tf.summary.scalar('hold_loss', self.c_hd)
            loss_summary = tf.summary.scalar('test_loss', self.loss)
            # self.summary_op_trg = tf.summary.merge([loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summary_op_trg = tf.summary.merge_all()


        


class End2End_v2(object):


    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.name='v2'


    def build_model(self):
        
        if self.mode == 'pretrain':
            pass

        elif self.mode == 'train' or  self.mode == 'eval':

            self.cat_dim = len(CAT_FEA_HOT)
            self.vlt_dim = len(VLT_FEA)
            self.sf_dim = len(SF_FEA)
            self.oth_dim = len(MORE_FEA)
            self.is_dim = len(IS_FEA)
            self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim

            self.hidden_dim = [[20, 20], 8]

            self.output_dim = 1
            self.q = 0.9

            with tf.name_scope('Data'):
                self.x_vlt = tf.placeholder(tf.float32, shape=[None, self.vlt_dim], name='Input_vlt')
                self.x_sf = tf.placeholder(tf.float32, shape=[None, self.sf_dim], name='Input_sf')
                self.x_cat = tf.placeholder(tf.float32, shape=[None, self.cat_dim], name='Input_pf')
                self.x_oth = tf.placeholder(tf.float32, shape=[None, self.oth_dim], name='Input_more')
                self.x_is = tf.placeholder(tf.float32, shape=[None, self.is_dim], name='Input_IS')
                self.mean_vlt = tf.expand_dims(self.x_vlt[:,-17],1, name='mean_vlt')
                self.review_p = tf.expand_dims(self.x_oth[:,0],1, name='review_p')


            with tf.name_scope('Label'):
                self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='Label')


            with tf.variable_scope('Layer_1_vlt'):
                self.W1_vlt = tf.Variable(tf.truncated_normal([self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]], 
                                                                stddev=0.001), name='Weight_1_vlt')
                self.b1_vlt = tf.Variable(tf.zeros([self.hidden_dim[0][0]]), name='Bias_1_vlt')
                self.l1_vlt = tf.add(tf.matmul(tf.concat([self.x_vlt, self.x_cat], axis=1), self.W1_vlt), self.b1_vlt)
                self.l1_vlt = tf.nn.relu(self.l1_vlt)


            with tf.variable_scope('Layer_1_sf'):
                self.W1_sf = tf.Variable(tf.truncated_normal([self.sf_dim+self.cat_dim, self.hidden_dim[0][1]], 
                                                                stddev=0.001), name='Weight_1_sf')
                self.b1_sf = tf.Variable(tf.zeros([self.hidden_dim[0][1]]), name='Bias_1_sf')
                self.l1_sf = tf.add(tf.matmul(tf.concat([self.x_sf, self.x_cat], axis=1), self.W1_sf), self.b1_sf)
                self.l1_sf = tf.nn.relu(self.l1_sf)
                        
            # with tf.variable_scope('Layer_1_profile'):
            #     W1_pf = tf.Variable(tf.truncated_normal([cat_dim, hidden_dim[0][2]], stddev=0.001), name='Weight_1_pf')
            #     b1_pf = tf.Variable(tf.zeros([hidden_dim[0][2]]), name='Bias_1_pf')
            #     l1_pf = tf.add(tf.matmul(x_cat, W1_pf), b1_pf)
            #     l1_pf = tf.nn.relu(l1_pf)

            with tf.variable_scope('Layer_2'):
                self.W2 = tf.Variable(tf.truncated_normal([self.hidden_dim[0][0]+self.hidden_dim[0][1]+self.oth_dim, 
                                                            self.hidden_dim[1]], stddev=0.001), name='Weight_2')
                self.b2 = tf.Variable(tf.zeros([self.hidden_dim[1]]), name='Bias_3')
                self.l2 = tf.add(tf.matmul(tf.concat([self.l1_vlt, self.l1_sf, self.x_oth], axis=1), self.W2), self.b2)
                self.l2 = tf.nn.relu(self.l2)


            with tf.variable_scope('Layer_3'):
                self.W3 = tf.Variable(tf.truncated_normal([self.hidden_dim[1], self.output_dim], stddev=0.001), name='Weight_3')
                self.b3 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_3')
                self.output = tf.add(tf.matmul(tf.concat([self.l2], axis=1), self.W3), self.b3)
                self.error = self.y - self.output

            with tf.variable_scope('Loss'):
            #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
                self.c_os = self.q * self.error
                self.c_hd = (self.q-1) * self.error
                self.loss = tf.reduce_mean(tf.maximum(self.c_os, self.c_hd) )
            #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
            #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
            #     loss = tf.reduce_mean(c_os+ c_hd )    

            with tf.name_scope('Optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
            # back_summary = tf.summary.scalar('back_loss', self.c_os)
            # hold_summary = tf.summary.scalar('hold_loss', self.c_hd)
            loss_summary = tf.summary.scalar('test_loss', self.loss)
            # self.summary_op_trg = tf.summary.merge([loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summary_op_trg = tf.summary.merge_all()


        

class End2End_v3(object):


    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.name='v3'


    def build_model(self):
        
        if self.mode == 'pretrain':
            pass

        elif self.mode == 'train' or  self.mode == 'eval':

            self.cat_dim = len(CAT_FEA_HOT)
            self.vlt_dim = len(VLT_FEA)
            self.sf_dim = len(SF_FEA)
            self.oth_dim = len(MORE_FEA)
            self.is_dim = len(IS_FEA)
            self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim

            self.hidden_dim = [[20, 20], 20, 8]

            self.output_dim = 1
            self.q = 0.9

            with tf.name_scope('Data'):
                self.x_vlt = tf.placeholder(tf.float32, shape=[None, self.vlt_dim], name='Input_vlt')
                self.x_sf = tf.placeholder(tf.float32, shape=[None, self.sf_dim], name='Input_sf')
                self.x_cat = tf.placeholder(tf.float32, shape=[None, self.cat_dim], name='Input_pf')
                self.x_oth = tf.placeholder(tf.float32, shape=[None, self.oth_dim], name='Input_more')
                self.x_is = tf.placeholder(tf.float32, shape=[None, self.is_dim], name='Input_IS')
                self.mean_vlt = tf.expand_dims(self.x_vlt[:,-17],1, name='mean_vlt')
                self.review_p = tf.expand_dims(self.x_oth[:,0],1, name='review_p')

            with tf.name_scope('Label'):
                self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='Label')


            with tf.variable_scope('Layer_1_vlt'):
                self.W1_vlt = tf.Variable(tf.truncated_normal([self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]], 
                                                                stddev=0.001), name='Weight_1_vlt')
                self.b1_vlt = tf.Variable(tf.zeros([self.hidden_dim[0][0]]), name='Bias_1_vlt')
                self.l1_vlt = tf.add(tf.matmul(tf.concat([self.x_vlt, self.x_cat], axis=1), self.W1_vlt), self.b1_vlt)
                self.l1_vlt = tf.nn.relu(self.l1_vlt)

            with tf.variable_scope('Layer_1_sf'):
                self.W1_sf = tf.Variable(tf.truncated_normal([self.sf_dim+self.cat_dim, self.hidden_dim[0][1]], 
                                                                stddev=0.001), name='Weight_1_sf')
                self.b1_sf = tf.Variable(tf.zeros([self.hidden_dim[0][1]]), name='Bias_1_sf')
                self.l1_sf = tf.add(tf.matmul(tf.concat([self.x_sf, self.x_cat], axis=1), self.W1_sf), self.b1_sf)
                self.l1_sf = tf.nn.relu(self.l1_sf)
                        
            # with tf.variable_scope('Layer_1_profile'):
            #     W1_pf = tf.Variable(tf.truncated_normal([cat_dim, hidden_dim[0][2]], stddev=0.001), name='Weight_1_pf')
            #     b1_pf = tf.Variable(tf.zeros([hidden_dim[0][2]]), name='Bias_1_pf')
            #     l1_pf = tf.add(tf.matmul(x_cat, W1_pf), b1_pf)
            #     l1_pf = tf.nn.relu(l1_pf)

            with tf.variable_scope('Layer_2'):
                self.W2 = tf.Variable(tf.truncated_normal([self.hidden_dim[0][0]+self.hidden_dim[0][1]+self.oth_dim, 
                                                            self.hidden_dim[1]], stddev=0.001), name='Weight_2')
                self.b2 = tf.Variable(tf.zeros([self.hidden_dim[1]]), name='Bias_2')
                self.l2 = tf.add(tf.matmul(tf.concat([self.l1_vlt, self.l1_sf, self.x_oth], axis=1), self.W2), self.b2)
                self.l2 = tf.nn.relu(self.l2)

            with tf.variable_scope('Layer_3'):
                self.W3 = tf.Variable(tf.truncated_normal([self.hidden_dim[1], self.hidden_dim[2]], stddev=0.001), name='Weight_3')
                self.b3 = tf.Variable(tf.zeros([self.hidden_dim[2]]), name='Bias_3')
                self.l3 = tf.add(tf.matmul(tf.concat([self.l2], axis=1), self.W3), self.b3)
                self.l3 = tf.nn.relu(self.l3)

            with tf.variable_scope('Layer_final'):
                self.W4 = tf.Variable(tf.truncated_normal([self.hidden_dim[2], self.output_dim], stddev=0.001), name='Weight_4')
                self.b4 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_4')
                self.output = tf.add(tf.matmul(tf.concat([self.l3, self.x_is], axis=1), self.W4), self.b4)
                self.error = self.y - self.output


            with tf.variable_scope('Loss'):
            #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
                self.c_os = self.q * self.error
                self.c_hd = (self.q-1) * self.error
                self.loss = tf.reduce_mean(tf.maximum(self.c_os, self.c_hd) )
            #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
            #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
            #     loss = tf.reduce_mean(c_os+ c_hd )    

            with tf.name_scope('Optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
            # back_summary = tf.summary.scalar('back_loss', self.c_os)
            # hold_summary = tf.summary.scalar('hold_loss', self.c_hd)
            loss_summary = tf.summary.scalar('test_loss', self.loss)
            # self.summary_op_trg = tf.summary.merge([loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summary_op_trg = tf.summary.merge_all()


        


class End2End_v4(object):


    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.name='v4'


    def build_model(self):
        
        if self.mode == 'pretrain':
            pass

        elif self.mode == 'train' or  self.mode == 'eval':

            self.cat_dim = len(CAT_FEA_HOT)
            self.vlt_dim = len(VLT_FEA)
            self.sf_dim = len(SF_FEA)
            self.oth_dim = len(MORE_FEA)
            self.is_dim = len(IS_FEA)
            self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim

            self.hidden_dim = [[20, 20], [5, 5], [1, 1]]

            self.output_dim = 1
            self.q = 0.9

            with tf.name_scope('Data'):
                self.x_vlt = tf.placeholder(tf.float32, shape=[None, self.vlt_dim], name='Input_vlt')
                self.x_sf = tf.placeholder(tf.float32, shape=[None, self.sf_dim], name='Input_sf')
                self.x_cat = tf.placeholder(tf.float32, shape=[None, self.cat_dim], name='Input_pf')
                self.x_oth = tf.placeholder(tf.float32, shape=[None, self.oth_dim], name='Input_more')
                self.x_is = tf.placeholder(tf.float32, shape=[None, self.is_dim], name='Input_IS')
                self.mean_vlt = tf.expand_dims(self.x_vlt[:,-17],1, name='mean_vlt')
                self.review_p = tf.expand_dims(self.x_oth[:,0],1, name='review_p')

            with tf.name_scope('Label'):
                self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='Label')
                self.y_vlt = tf.placeholder(tf.float32, shape=[None, 1], name='Label_vlt')
                self.y_sf = tf.placeholder(tf.float32, shape=[None, 1], name='Label_sf')


            with tf.variable_scope('Layer_1_vlt'):
                self.W1_vlt = tf.Variable(tf.truncated_normal([self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]], 
                                                                stddev=0.001), name='Weight_1_vlt')
                self.b1_vlt = tf.Variable(tf.zeros([self.hidden_dim[0][0]]), name='Bias_1_vlt')
                self.l1_vlt = tf.add(tf.matmul(tf.concat([self.x_vlt, self.x_cat], axis=1), self.W1_vlt), self.b1_vlt)
                self.l1_vlt = tf.nn.relu(self.l1_vlt)


            with tf.variable_scope('Layer_1_sf'):
                self.W1_sf = tf.Variable(tf.truncated_normal([self.sf_dim+self.cat_dim, self.hidden_dim[0][1]], 
                                                                stddev=0.001), name='Weight_1_sf')
                self.b1_sf = tf.Variable(tf.zeros([self.hidden_dim[0][1]]), name='Bias_1_sf')
                self.l1_sf = tf.add(tf.matmul(tf.concat([self.x_sf, self.x_cat], axis=1), self.W1_sf), self.b1_sf)
                self.l1_sf = tf.nn.relu(self.l1_sf)
                        
            # with tf.variable_scope('Layer_1_profile'):
            #     W1_pf = tf.Variable(tf.truncated_normal([cat_dim, hidden_dim[0][2]], stddev=0.001), name='Weight_1_pf')
            #     b1_pf = tf.Variable(tf.zeros([hidden_dim[0][2]]), name='Bias_1_pf')
            #     l1_pf = tf.add(tf.matmul(x_cat, W1_pf), b1_pf)
            #     l1_pf = tf.nn.relu(l1_pf)

            with tf.variable_scope('Layer_2_vlt'):
                self.W2_vlt = tf.Variable(tf.truncated_normal([self.hidden_dim[0][0], 
                                                            self.hidden_dim[1][0]], stddev=0.001), name='Weight_2_vlt')
                self.b2_vlt = tf.Variable(tf.zeros([self.hidden_dim[1][0]]), name='Bias_2_vlt')
                self.l2_vlt = tf.add(tf.matmul(self.l1_vlt, self.W2_vlt), self.b2_vlt)
                self.l2_vlt = tf.nn.relu(self.l2_vlt)


            with tf.variable_scope('Layer_2_sf'):
                self.W2_sf = tf.Variable(tf.truncated_normal([self.hidden_dim[0][1], 
                                                            self.hidden_dim[1][1]], stddev=0.001), name='Weight_2_sf')
                self.b2_sf = tf.Variable(tf.zeros([self.hidden_dim[1][1]]), name='Bias_2_sf')
                self.l2_sf = tf.add(tf.matmul(self.l1_sf, self.W2_sf), self.b2_sf)
                self.l2_sf = tf.nn.relu(self.l2_sf)


            with tf.variable_scope('Layer_3_vlt'):
                self.W3_vlt = tf.Variable(tf.truncated_normal([self.hidden_dim[1][0], 
                                                            self.hidden_dim[2][0]], stddev=0.001), name='Weight_3_vlt')
                self.b3_vlt = tf.Variable(tf.zeros([self.hidden_dim[2][0]]), name='Bias_3_vlt')
                self.l3_vlt = tf.add(tf.matmul(self.l2_vlt, self.W3_vlt), self.b3_vlt)
                # self.l3_vlt = tf.nn.relu(self.l3_vlt)
                self.l3_vlt_t = self.l3_vlt / pd_scaler.loc[1, 'vlt_actual'] + pd_scaler.loc[0, 'vlt_actual'] 
                rp_t = self.review_p / pd_scaler.loc[1, 'review_period'] + pd_scaler.loc[0, 'review_period'] 
                self.l3_vlt_rp_t = tf.log(self.l3_vlt_t+rp_t+1)


            with tf.variable_scope('Layer_3_sf'):
                self.W3_sf = tf.Variable(tf.truncated_normal([self.hidden_dim[1][1], 
                                                         self.hidden_dim[2][1]], stddev=0.001), name='Weight_3_sf')
                self.b3_sf = tf.Variable(tf.zeros([self.hidden_dim[2][1]]), name='Bias_3_sf')
                self.l3_sf = tf.add(tf.matmul(self.l2_sf, self.W3_sf), self.b3_sf)
                # self.l3_sf = tf.nn.relu(self.l3_sf)
                self.l3_sf_t = tf.log(self.l3_sf / pd_scaler.loc[1, 'label_sf'] + pd_scaler.loc[0, 'label_sf'] + 1)


            with tf.variable_scope('Layer_final'):
                self.W4 = tf.Variable(tf.truncated_normal([self.hidden_dim[2][0]+self.hidden_dim[2][1]+self.is_dim, 
                                            self.output_dim], stddev=0.001), name='Weight_4')
                self.b4 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_4') # need bias here?
                self.output_t = tf.add(tf.matmul(tf.concat([self.l3_sf_t, self.l3_vlt_rp_t, self.x_is], axis=1), self.W4), self.b4)
                self.output_t = tf.exp(self.output_t) - 1
                self.output = (self.output_t - pd_scaler.loc[0, 'demand_RV']) * pd_scaler.loc[1, 'demand_RV']
                self.error = self.y - self.output
                self.error_vlt = self.y_vlt - self.l3_vlt 
                self.error_sf = self.y_sf - self.l3_sf


            with tf.variable_scope('Loss'):
            #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
                self.c_os = self.q * self.error
                self.c_hd = (self.q-1) * self.error
                self.loss = tf.reduce_mean(tf.maximum(self.c_os, self.c_hd) 
                                + 0.5*tf.square(self.error_vlt)+ 0.5*tf.square(self.error_sf) )
            #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
            #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
            #     loss = tf.reduce_mean(c_os+ c_hd )    

            with tf.name_scope('Optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
            # back_summary = tf.summary.scalar('back_loss', self.c_os)
            # hold_summary = tf.summary.scalar('hold_loss', self.c_hd)
            loss_summary = tf.summary.scalar('test_loss', self.loss)
            # self.summary_op_trg = tf.summary.merge([loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summary_op_trg = tf.summary.merge_all()





class End2End_v5(object):


    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.name='v5'


    def build_model(self):
        
        if self.mode == 'pretrain':
            pass

        elif self.mode == 'train' or  self.mode == 'eval':

            self.cat_dim = len(CAT_FEA_HOT)
            self.vlt_dim = len(VLT_FEA)
            self.sf_dim = len(SF_FEA)
            self.oth_dim = len(MORE_FEA)
            self.is_dim = len(IS_FEA)
            self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim

            self.hidden_dim = [[20, 20], [5, 5], [1, 1], 3]

            self.output_dim = 1
            self.q = 0.9

            with tf.name_scope('Data'):
                self.x_vlt = tf.placeholder(tf.float32, shape=[None, self.vlt_dim], name='Input_vlt')
                self.x_sf = tf.placeholder(tf.float32, shape=[None, self.sf_dim], name='Input_sf')
                self.x_cat = tf.placeholder(tf.float32, shape=[None, self.cat_dim], name='Input_pf')
                self.x_oth = tf.placeholder(tf.float32, shape=[None, self.oth_dim], name='Input_more')
                self.x_is = tf.placeholder(tf.float32, shape=[None, self.is_dim], name='Input_IS')
                self.mean_vlt = tf.expand_dims(self.x_vlt[:,-17],1, name='mean_vlt')
                self.review_p = tf.expand_dims(self.x_oth[:,0],1, name='review_p')

            with tf.name_scope('Label'):
                self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='Label')
                self.y_vlt = tf.placeholder(tf.float32, shape=[None, 1], name='Label_vlt')
                self.y_sf = tf.placeholder(tf.float32, shape=[None, 1], name='Label_sf')


            with tf.variable_scope('Layer_1_vlt'):
                self.W1_vlt = tf.Variable(tf.truncated_normal([self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]], 
                                                                stddev=0.001), name='Weight_1_vlt')
                self.b1_vlt = tf.Variable(tf.zeros([self.hidden_dim[0][0]]), name='Bias_1_vlt')
                self.l1_vlt = tf.add(tf.matmul(tf.concat([self.x_vlt, self.x_cat], axis=1), self.W1_vlt), self.b1_vlt)
                self.l1_vlt = tf.nn.relu(self.l1_vlt)


            with tf.variable_scope('Layer_1_sf'):
                self.W1_sf = tf.Variable(tf.truncated_normal([self.sf_dim+self.cat_dim, self.hidden_dim[0][1]], 
                                                                stddev=0.001), name='Weight_1_sf')
                self.b1_sf = tf.Variable(tf.zeros([self.hidden_dim[0][1]]), name='Bias_1_sf')
                self.l1_sf = tf.add(tf.matmul(tf.concat([self.x_sf, self.x_cat], axis=1), self.W1_sf), self.b1_sf)
                self.l1_sf = tf.nn.relu(self.l1_sf)
                        
            # with tf.variable_scope('Layer_1_profile'):
            #     W1_pf = tf.Variable(tf.truncated_normal([cat_dim, hidden_dim[0][2]], stddev=0.001), name='Weight_1_pf')
            #     b1_pf = tf.Variable(tf.zeros([hidden_dim[0][2]]), name='Bias_1_pf')
            #     l1_pf = tf.add(tf.matmul(x_cat, W1_pf), b1_pf)
            #     l1_pf = tf.nn.relu(l1_pf)

            with tf.variable_scope('Layer_2_vlt'):
                self.W2_vlt = tf.Variable(tf.truncated_normal([self.hidden_dim[0][0], 
                                                            self.hidden_dim[1][0]], stddev=0.001), name='Weight_2_vlt')
                self.b2_vlt = tf.Variable(tf.zeros([self.hidden_dim[1][0]]), name='Bias_2_vlt')
                self.l2_vlt = tf.add(tf.matmul(self.l1_vlt, self.W2_vlt), self.b2_vlt)
                self.l2_vlt = tf.nn.relu(self.l2_vlt)


            with tf.variable_scope('Layer_2_sf'):
                self.W2_sf = tf.Variable(tf.truncated_normal([self.hidden_dim[0][1], 
                                                            self.hidden_dim[1][1]], stddev=0.001), name='Weight_2_sf')
                self.b2_sf = tf.Variable(tf.zeros([self.hidden_dim[1][1]]), name='Bias_2_sf')
                self.l2_sf = tf.add(tf.matmul(self.l1_sf, self.W2_sf), self.b2_sf)
                self.l2_sf = tf.nn.relu(self.l2_sf)


            with tf.variable_scope('Layer_3_vlt'):
                self.W3_vlt = tf.Variable(tf.truncated_normal([self.hidden_dim[1][0], 
                                                            self.hidden_dim[2][0]], stddev=0.001), name='Weight_3_vlt')
                self.b3_vlt = tf.Variable(tf.zeros([self.hidden_dim[2][0]]), name='Bias_3_vlt')
                self.l3_vlt = tf.add(tf.matmul(self.l2_vlt, self.W3_vlt), self.b3_vlt)
                # self.l3_vlt = tf.nn.relu(self.l3_vlt)
                self.l3_vlt_t = self.l3_vlt / pd_scaler.loc[1, 'vlt_actual'] + pd_scaler.loc[0, 'vlt_actual'] 
                rp_t = self.review_p / pd_scaler.loc[1, 'review_period'] + pd_scaler.loc[0, 'review_period'] 
                self.l3_vlt_rp_t = tf.log(self.l3_vlt_t+rp_t+1)


            with tf.variable_scope('Layer_3_sf'):
                self.W3_sf = tf.Variable(tf.truncated_normal([self.hidden_dim[1][1], 
                                                         self.hidden_dim[2][1]], stddev=0.001), name='Weight_3_sf')
                self.b3_sf = tf.Variable(tf.zeros([self.hidden_dim[2][1]]), name='Bias_3_sf')
                self.l3_sf = tf.add(tf.matmul(self.l2_sf, self.W3_sf), self.b3_sf)
                # self.l3_sf = tf.nn.relu(self.l3_sf)
                self.l3_sf_t = tf.log(self.l3_sf / pd_scaler.loc[1, 'label_sf'] + pd_scaler.loc[0, 'label_sf'] + 1)


            # with tf.variable_scope('Layer_final'):
            #     self.W4 = tf.Variable(tf.truncated_normal([self.hidden_dim[2][0]+self.hidden_dim[2][1]+self.is_dim, 
            #                                 self.output_dim], stddev=0.001), name='Weight_4')
            #     self.b4 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_4') # need bias here?
            #     self.output_t = tf.add(tf.matmul(tf.concat([self.l3_sf_t, self.l3_vlt_rp_t, self.x_is], axis=1), self.W4), self.b4)
            #     self.output_t = tf.exp(self.output_t) - 1
            #     self.output = (self.output_t - pd_scaler.loc[0, 'demand_RV']) * pd_scaler.loc[1, 'demand_RV']
            #     self.error = self.y - self.output
            #     self.error_vlt = self.y_vlt - self.l3_vlt 
            #     self.error_sf = self.y_sf - self.l3_sf


            with tf.variable_scope('Layer_3'):
                self.W3 = tf.Variable(tf.truncated_normal([self.hidden_dim[1][0]+self.hidden_dim[1][1]+self.oth_dim+self.is_dim, 
                                            self.hidden_dim[3]], stddev=0.001), name='Weight_3')
                self.b3 = tf.Variable(tf.zeros([self.hidden_dim[3]]), name='Bias_3') # need bias here?
                self.l3 = tf.add(tf.matmul(tf.concat([self.l2_sf, self.l2_vlt, self.x_oth, self.x_is], axis=1), self.W3), self.b3)
                self.l3 = tf.nn.relu(self.l3)


            with tf.variable_scope('Layer_final'):
                self.W4 = tf.Variable(tf.truncated_normal([self.hidden_dim[3], self.output_dim], stddev=0.001), name='Weight_4')
                self.b4 = tf.Variable(tf.zeros([self.output_dim]), name='Bias_4') # need bias here?
                self.output = tf.add(tf.matmul(self.l3, self.W4), self.b4)
                self.error = self.y - self.output
                self.error_vlt = self.y_vlt - self.l3_vlt 
                self.error_sf = self.y_sf - self.l3_sf


            with tf.variable_scope('Loss'):
            #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
                self.c_os = self.q * self.error
                self.c_hd = (self.q-1) * self.error
                self.loss = tf.reduce_mean(tf.maximum(self.c_os, self.c_hd) 
                                + 0.5*tf.square(self.error_vlt)+ 0.5*tf.square(self.error_sf) )
            #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
            #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
            #     loss = tf.reduce_mean(c_os+ c_hd )    

            with tf.name_scope('Optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
            # back_summary = tf.summary.scalar('back_loss', self.c_os)
            # hold_summary = tf.summary.scalar('hold_loss', self.c_hd)
            loss_summary = tf.summary.scalar('test_loss', self.loss)
            # self.summary_op_trg = tf.summary.merge([loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summary_op_trg = tf.summary.merge_all()




class End2End_v5_tc(nn.Module):


    def __init__(self, device, tf_graph=False):
        
        super(End2End_v5_tc, self).__init__()
        self.name='v5_tc'
        self.device = device
        self.tf_graph = tf_graph


        self.cat_dim = len(CAT_FEA_HOT)
        self.vlt_dim = len(VLT_FEA)
        self.sf_dim = len(SF_FEA)
        self.oth_dim = len(MORE_FEA)
        self.is_dim = len(IS_FEA)

        self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim
        self.hidden_dim = [[100, 120], [1, 1], 100, 30]
        self.output_dim = 1
        self.q = 0.9

        self.fc_vlt_1 = nn.Linear(self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]) 
        self.fc_vlt_2 = nn.Linear(self.hidden_dim[0][0], self.hidden_dim[1][0])  

        self.fc_sf_1 = nn.Linear(self.sf_dim+self.cat_dim, self.hidden_dim[0][1]) 
        self.fc_sf_2 = nn.Linear(self.hidden_dim[0][1], self.hidden_dim[1][1]) 

        self.fc_3 = nn.Linear(self.hidden_dim[0][0]+self.hidden_dim[0][1]+self.oth_dim+self.is_dim, 
                                            self.hidden_dim[2])
        self.fc_4 = nn.Linear(self.hidden_dim[2], self.hidden_dim[3])
        self.fc_5 = nn.Linear(self.hidden_dim[3], self.output_dim)
        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        self.fc_vlt_1.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_1.bias.data.fill_(0)
        self.fc_vlt_2.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_2.bias.data.fill_(0)

        self.fc_sf_1.weight.data.normal_(0.0, 0.01)
        self.fc_sf_1.bias.data.fill_(0)
        self.fc_sf_2.weight.data.normal_(0.0, 0.01)
        self.fc_sf_2.bias.data.fill_(0)

        self.fc_3.weight.data.normal_(0.0, 0.01)
        self.fc_3.bias.data.fill_(0)
        self.fc_4.weight.data.normal_(0.0, 0.01)
        self.fc_4.bias.data.fill_(0)
        self.fc_5.weight.data.normal_(0.0, 0.01)
        self.fc_5.bias.data.fill_(0)


    def forward(self, x_vlt, x_sf, x_cat, x_oth, x_is):

        x1 = self.fc_vlt_1(torch.cat([x_vlt, x_cat], 1))
        x1 = F.relu(x1)
        o_vlt = self.fc_vlt_2(x1)

        x2 = self.fc_sf_1(torch.cat([x_sf, x_cat], 1))
        x2 = F.relu(x2)
        o_sf = self.fc_sf_2(x2)

        x = self.fc_3(torch.cat([x1, x2, x_oth, x_is],1))
        x = F.relu(x)
        x = self.fc_4(x)
        x = F.relu(x)
        x = self.fc_5(x)

        return x, o_vlt, o_sf



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
        context_vectors = torch.sigmoid( self.global_mlp(hidden_future_concat) )
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

    
    def forward(self, x_seq_hist, y_seq_hist, x_seq_pred):
        
        bsize = x_seq_hist.size(0)

        assert x_seq_hist.size(1) == self.hist_long
        assert x_seq_hist.size(2) == self.input_dim
        assert x_seq_pred.size(1) == self.pred_long

        x_feat_hist = torch.tanh(self.linear_encoder(x_seq_hist))
        x_feat_pred = torch.tanh(self.linear_encoder(x_seq_pred))
        
        
        self.lstm.flatten_parameters()
        
        y_seq_hist_1 = y_seq_hist.view(y_seq_hist.size(0), y_seq_hist.size(1), 1)

        x_total_hist =  torch.cat([x_feat_hist,y_seq_hist_1], dim=2)
        x_total_pred =  torch.cat([x_feat_pred], dim=2)
                
        #print('xy seq hist size ', xy_seq_hist.size())

        hiddens, (ht, c) = self.lstm(x_total_hist)
        #print('LSTM total output hidden size ', hiddens.size())
        # print('LSTM last output hidden size ', ht.size())
        # print('LSTM last output hidden size ', x_total_pred.size())
        
        ht = ht.view(ht.size(1),ht.size(2))

        result = self.decoder(ht, x_total_pred)
        result = result.view(-1, rnn_pred_long*num_quantiles)

        return result






class End2End_v6_tc(nn.Module):


    def __init__(self, device, tf_graph=False):
        
        super(End2End_v6_tc, self).__init__()
        self.name='v6_tc'
        self.device = device
        self.tf_graph = tf_graph

        self.cat_dim = len(CAT_FEA_HOT)
        self.vlt_dim = len(VLT_FEA)
        self.sf_dim = len(SF_FEA)
        self.oth_dim = len(MORE_FEA)
        self.is_dim = len(IS_FEA)

        self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim
        self.hidden_dim = [[50, 50], [20, 20], [1, 1], 10]
        self.output_dim = 1
        self.q = 0.9

        self.fc_vlt_1 = nn.Linear(self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]) 
        self.fc_vlt_2 = nn.Linear(self.hidden_dim[0][0], self.hidden_dim[1][0])  
        self.fc_vlt_3 = nn.Linear(self.hidden_dim[1][0], self.hidden_dim[2][0])  

        self.sf_mqrnn = MQ_RNN(rnn_input_dim, rnn_hidden_len, rnn_cxt_len, num_quantiles, rnn_pred_long, rnn_hist_long)

        self.fc_3 = nn.Linear(rnn_pred_long*num_quantiles + self.hidden_dim[1][1]+self.oth_dim+self.is_dim, 
                                            self.hidden_dim[3])
        self.fc_4 = nn.Linear(self.hidden_dim[3], self.output_dim)
        self.init_weights()


    def init_weights(self):

        self.fc_vlt_1.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_1.bias.data.fill_(0)
        self.fc_vlt_2.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_2.bias.data.fill_(0)
        self.fc_vlt_3.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_3.bias.data.fill_(0)

        # self.sf_mqrnn.load_state_dict(torch.load('../logs/torch/mqrnn_35.pkl', map_location=self.device))
        # for param in self.sf_mqrnn.parameters():
        #     param.requires_grad = False

        self.fc_3.weight.data.normal_(0.0, 0.01)
        self.fc_3.bias.data.fill_(0)
        self.fc_4.weight.data.normal_(0.0, 0.01)
        self.fc_4.bias.data.fill_(0)


    def forward(self, enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is):

        x1 = self.fc_vlt_1(torch.cat([x_vlt, x_cat], 1))
        x1 = F.relu(x1)
        x1 = self.fc_vlt_2(x1)
        x1 = F.relu(x1)
        o_vlt = self.fc_vlt_3(x1)

        o_sf = self.sf_mqrnn(enc_X, enc_y, dec_X)

        x = self.fc_3(torch.cat([x1, o_sf, x_oth, x_is],1))
        x = F.relu(x)
        x = self.fc_4(x)

        return x, o_vlt, o_sf




class End2End_v7_tc(nn.Module):


    def __init__(self, device, tf_graph=False):
        
        super(End2End_v7_tc, self).__init__()
        self.name='v7_tc'
        self.device = device
        self.tf_graph = tf_graph

        self.cat_dim = len(CAT_FEA_HOT)
        self.vlt_dim = len(VLT_FEA)
        self.sf_dim = len(SF_FEA)
        self.oth_dim = len(MORE_FEA)
        self.is_dim = len(IS_FEA)

        self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim
        self.hidden_dim = [[50, None], [20, None], [1, None], [rnn_pred_long, None], 10]
        self.output_dim = 1
        self.q = 0.9

        self.fc_vlt_1 = nn.Linear(self.vlt_dim+self.cat_dim, self.hidden_dim[0][0]) 
        self.fc_vlt_2 = nn.Linear(self.hidden_dim[0][0], self.hidden_dim[1][0])  
        self.fc_vlt_3 = nn.Linear(self.hidden_dim[1][0], self.hidden_dim[2][0])  

        if self.tf_graph:
            self.sf_mqrnn = nn.Linear(rnn_hist_long, rnn_pred_long*num_quantiles).to(device)
        else:
            self.sf_mqrnn = MQ_RNN(rnn_input_dim, rnn_hidden_len, rnn_cxt_len, num_quantiles, rnn_pred_long, rnn_hist_long)

        self.fc_vlt_aug = nn.Linear(self.hidden_dim[2][0]+self.oth_dim, self.hidden_dim[3][0])

        self.fc_3 = {}
        for i in range(self.hidden_dim[3][0]):
            # self.fc_3[i] = nn.Linear(rnn_pred_long*num_quantiles + 1, 1)
            self.fc_3[i] = nn.Linear(num_quantiles + 1, 1).to(self.device)

        self.fc_4 = nn.Linear(self.hidden_dim[3][0], self.hidden_dim[4])
        self.fc_5 = nn.Linear(self.hidden_dim[4] + self.is_dim, self.output_dim)
        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        self.fc_vlt_1.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_1.bias.data.fill_(0)
        self.fc_vlt_2.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_2.bias.data.fill_(0)
        self.fc_vlt_3.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_3.bias.data.fill_(0)

        self.sf_mqrnn.load_state_dict(torch.load('../logs/torch/mqrnn_35.pkl', map_location=self.device))
        for param in self.sf_mqrnn.parameters():
            param.requires_grad = False

        self.fc_vlt_aug.weight.data.normal_(0.0, 0.01)
        self.fc_vlt_aug.bias.data.fill_(0)
        self.fc_3.weight.data.normal_(0.0, 0.01)
        self.fc_3.bias.data.fill_(0)
        self.fc_4.weight.data.normal_(0.0, 0.01)
        self.fc_4.bias.data.fill_(0)
        self.fc_5.weight.data.normal_(0.0, 0.01)
        self.fc_5.bias.data.fill_(0)



    def forward(self, enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is):

        x1 = self.fc_vlt_1(torch.cat([x_vlt, x_cat], 1))
        x1 = F.relu(x1)
        x1 = self.fc_vlt_2(x1)
        x1 = F.relu(x1)
        o_vlt = self.fc_vlt_3(x1)

        vlt_aug = self.fc_vlt_aug(torch.cat([o_vlt, x_oth], 1))
        if self.tf_graph:
            o_sf = self.sf_mqrnn(enc_X).view(-1, rnn_pred_long, num_quantiles)
        else:
            o_sf = self.sf_mqrnn(enc_X, enc_y, dec_X)

        sf_vlt = torch.Tensor(x1.shape[0], self.hidden_dim[3][0]).to(self.device)
        for i in range(self.hidden_dim[3][0]):
            vlt_1 = vlt_aug[:,i].view(-1, 1)
            sf_vlt[:,i] = self.fc_3[i](torch.cat([o_sf[:,i,:], vlt_1], 1)).view(-1)

        x = self.fc_4(sf_vlt)
        x = F.relu(x)
        x = self.fc_5(torch.cat([x, x_is], 1))

        return x, o_vlt, o_sf