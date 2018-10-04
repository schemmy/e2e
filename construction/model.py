# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 15:45:51
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-04 11:19:51
# Template:
# https://github.com/yunjey/domain-transfer-network/blob/master/model.py

import tensorflow as tf
from config import *
import pandas as pd


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

            self.hidden_dim = [[20, 20], [5, 5], [1, 1], 5]

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
                                            self.output_dim], stddev=0.001), name='Weight_3')
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
