# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-09-18 13:22:20
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-01 15:21:48

# from base.base_model import BaseModel
import tensorflow as tf


class End2End_v1(BaseModel):
    def __init__(self, config):
        super(End2End_v1, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        
        self.is_training = tf.placeholder(tf.bool)

        cat_dim = len(CAT_FEA_HOT)
        vlt_dim = len(VLT_FEA)
        sf_dim = len(SF_FEA)
        oth_dim = len(MORE_FEA)
        is_dim = len(IS_FEA)
        input_dim =  vlt_dim + sf_dim + oth_dim + is_dim + cat_dim

        hidden_dim = [[100, 120], 160, 30]

        output_dim = 1
        q = 0.9


        tf.reset_default_graph()
        tf.set_random_seed(0)

        with tf.name_scope('Data'):
            x_vlt = tf.placeholder(tf.float32, shape=[None, vlt_dim], name='Input_vlt')
            x_sf = tf.placeholder(tf.float32, shape=[None, sf_dim], name='Input_sf')
            x_cat = tf.placeholder(tf.float32, shape=[None, cat_dim], name='Input_pf')
            x_oth = tf.placeholder(tf.float32, shape=[None, oth_dim], name='Input_more')
            x_is = tf.placeholder(tf.float32, shape=[None, is_dim], name='Input_IS')
            mean_vlt = tf.expand_dims(x_vlt[:,-17],1, name='mean_vlt')
            review_p = tf.expand_dims(x_oth[:,0],1, name='review_p')

        with tf.name_scope('Label'):
            y = tf.placeholder(tf.float32, shape=[None, 1], name='Label')

        with tf.variable_scope('Layer_1_vlt'):
            W1_vlt = tf.Variable(tf.truncated_normal([vlt_dim+cat_dim, hidden_dim[0][0]], stddev=0.001), name='Weight_1_vlt')
            b1_vlt = tf.Variable(tf.zeros([hidden_dim[0][0]]), name='Bias_1_vlt')
            l1_vlt = tf.add(tf.matmul(tf.concat([x_vlt, x_cat], axis=1), W1_vlt), b1_vlt)
            l1_vlt = tf.nn.relu(l1_vlt)

        with tf.variable_scope('Layer_1_sf'):
            W1_sf = tf.Variable(tf.truncated_normal([sf_dim+cat_dim, hidden_dim[0][1]], stddev=0.001), name='Weight_1_sf')
            b1_sf = tf.Variable(tf.zeros([hidden_dim[0][1]]), name='Bias_1_sf')
            l1_sf = tf.add(tf.matmul(tf.concat([x_sf, x_cat], axis=1), W1_sf), b1_sf)
            l1_sf = tf.nn.relu(l1_sf)
                    
        # with tf.variable_scope('Layer_1_profile'):
        #     W1_pf = tf.Variable(tf.truncated_normal([cat_dim, hidden_dim[0][2]], stddev=0.001), name='Weight_1_pf')
        #     b1_pf = tf.Variable(tf.zeros([hidden_dim[0][2]]), name='Bias_1_pf')
        #     l1_pf = tf.add(tf.matmul(x_cat, W1_pf), b1_pf)
        #     l1_pf = tf.nn.relu(l1_pf)

        with tf.variable_scope('Layer_2'):
            W2 = tf.Variable(tf.truncated_normal([hidden_dim[0][0]+hidden_dim[0][1]+oth_dim, hidden_dim[1]], stddev=0.001), name='Weight_2')
            b2 = tf.Variable(tf.zeros([hidden_dim[1]]), name='Bias_3')
            l2 = tf.add(tf.matmul(tf.concat([l1_vlt, l1_sf, x_oth], axis=1), W2), b2)
            l2 = tf.nn.relu(l2)

        with tf.variable_scope('Layer_3'):
            W3 = tf.Variable(tf.truncated_normal([hidden_dim[1], hidden_dim[2]], stddev=0.001), name='Weight_3')
            b3 = tf.Variable(tf.zeros([hidden_dim[2]]), name='Bias_3')
            l3 = tf.add(tf.matmul(tf.concat([l2], axis=1), W3), b3)
            l3 = tf.nn.relu(l3)

        with tf.variable_scope('Layer_final'):
            W4 = tf.Variable(tf.truncated_normal([hidden_dim[2], 1], stddev=0.001), name='Weight_4')
            b4 = tf.Variable(tf.zeros([1]), name='Bias_4')
            output = tf.add(tf.matmul(tf.concat([l3, x_is], axis=1), W4), b4)
            error = y - output

        with tf.name_scope('loss'):
        #     loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )
            c_os = q*error
            c_hd = (q-1)*error
            loss = tf.reduce_mean(tf.maximum(c_os, c_hd) )
        #     c_os = tf.maximum(tf.zeros([1], tf.float32), q*error)
        #     c_hd = tf.multiply((1-q)*output, review_p + mean_vlt )*0.5
        #     loss = tf.reduce_mean(c_os+ c_hd )    

        with tf.name_scope('Optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
