# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 15:45:51
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-03 16:55:30
# Template:
# https://github.com/yunjey/domain-transfer-network/blob/master/model.py

import tensorflow as tf
from config import *




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


        