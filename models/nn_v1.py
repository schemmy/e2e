# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-09-18 13:22:20
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-09-18 13:59:49

# from base.base_model import BaseModel
import tensorflow as tf


# class ExampleModel(BaseModel):
#     def __init__(self, config):
#         super(ExampleModel, self).__init__(config)
#         self.build_model()
#         self.init_saver()

#     def build_model(self):
        
#         self.is_training = tf.placeholder(tf.bool)

#         self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
#         self.y = tf.placeholder(tf.float32, shape=[None, 10])

#         # network architecture
#         d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
#         d2 = tf.layers.dense(d1, 10, name="dense2")

#         with tf.name_scope("loss"):
#             self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
#             self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
#                                                                                          global_step=self.global_step_tensor)
#             correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#     def init_saver(self):
#         # here you initialize the tensorflow saver that will be used in saving the checkpoints.
#         self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

VLT_FEA = [
    'uprc', 'contract_stk_prc', 'wt', 'width', 'height', 'calc_volume', 'len',
    'vlt_count', 'vlt_sum', 'vlt_min', 'vlt_max', 'vlt_mean', 'vlt_std',
    'qtty_sum', 'qtty_min', 'qtty_max', 'qtty_mean', 'qtty_std', 
    'amount_sum', 'amount_min', 'amount_max', 'amount_mean', 'amount_std', 
    'vlt_count_6mo', 'vlt_sum_6mo', 'vlt_min_6mo', 'vlt_max_6mo', 'vlt_mean_6mo', 'vlt_std_6mo',
    'vendor_vlt_count', 'vendor_vlt_sum', 'vendor_vlt_min', 'vendor_vlt_max', 'vendor_vlt_mean', 'vendor_vlt_std', 
    'vendor_vlt_count_6mo', 'vendor_vlt_sum_6mo', 'vendor_vlt_min_6mo', 
    'vendor_vlt_max_6mo', 'vendor_vlt_mean_6mo', 'vendor_vlt_std_6mo', 
    'vendor_qtty_sum', 'vendor_qtty_min', 'vendor_qtty_max', 
    'vendor_qtty_mean', 'vendor_qtty_std', 'vendor_amount_sum',
    'vendor_amount_min', 'vendor_amount_max', 'vendor_amount_mean']

SF_FEA = ['q_7', 'q_14', 'q_28', 'q_56',
       'q_112', 'mean_3', 'mean_7', 'mean_14', 'mean_28', 'mean_56',
       'mean_112', 'diff_140_mean', 'mean_140_decay', 'median_140', 'min_140',
       'max_140', 'std_140', 'diff_60_mean', 'mean_60_decay', 'median_60',
       'min_60', 'max_60', 'std_60', 'diff_30_mean', 'mean_30_decay',
       'median_30', 'min_30', 'max_30', 'std_30', 'diff_14_mean',
       'mean_14_decay', 'median_14', 'min_14', 'max_14', 'std_14',
       'diff_7_mean', 'mean_7_decay', 'median_7', 'min_7', 'max_7', 'std_7',
       'diff_3_mean', 'mean_3_decay', 'median_3', 'min_3', 'max_3', 'std_3',
       'has_sales_days_in_last_140', 'last_has_sales_day_in_last_140',
       'first_has_sales_day_in_last_140', 'has_sales_days_in_last_60',
       'last_has_sales_day_in_last_60', 'first_has_sales_day_in_last_60',
       'has_sales_days_in_last_30', 'last_has_sales_day_in_last_30',
       'first_has_sales_day_in_last_30', 'has_sales_days_in_last_14',
       'last_has_sales_day_in_last_14', 'first_has_sales_day_in_last_14',
       'has_sales_days_in_last_7', 'last_has_sales_day_in_last_7',
       'first_has_sales_day_in_last_7']
   
MORE_FEA =['initial_stock', 'review_period', 'normal', 'gamma', 'eq']



vlt_dim = len(VLT_FEA)
sf_dim = len(SF_FEA)
oth_dim = len(MORE_FEA)
input_dim = vlt_dim + sf_dim + oth_dim

hidden_dim = [[120,150], [40,50], 30, 10]

output_dim = 1
q = 0.9

tf.reset_default_graph()

with tf.name_scope('my_scope'):
	x = tf.placeholder(tf.float32, shape=[None, input_dim], name='Input')
	y = tf.placeholder(tf.float32, shape=[None, 1], name='Label')


with tf.variable_scope('Layer_1'):
	W1_vlt = tf.Variable(tf.truncated_normal([vlt_dim, hidden_dim[0][0]], stddev=0.001), name='Weight_1_vlt')
	b1_vlt = tf.Variable(tf.zeros([hidden_dim[0][0]]), name='Bias_1_vlt')
	l1_vlt = tf.add(tf.matmul(x[:,:vlt_dim], W1_vlt), b1_vlt)
	l1_vlt = tf.nn.relu(l1_vlt)

	W1_sf = tf.Variable(tf.truncated_normal([sf_dim, hidden_dim[0][1]], stddev=0.001), name='Weight_1_sf')
	b1_sf = tf.Variable(tf.zeros([hidden_dim[0][1]]), name='Weight_1_sf')
	l1_sf = tf.add(tf.matmul(x[:, vlt_dim:vlt_dim+sf_dim], W1_sf), b1_sf)
	l1_sf = tf.nn.relu(l1_sf)

with tf.variable_scope('Layer_2'):
	W2_vlt = tf.Variable(tf.truncated_normal([hidden_dim[0][0], hidden_dim[1][0]], stddev=0.001), name='Weight_2_vlt')
	b2_vlt = tf.Variable(tf.zeros([hidden_dim[1][0]]), name='Bias_2_vlt')
	l2_vlt = tf.add(tf.matmul(l1_vlt, W2_vlt), b2_vlt)
	l2_vlt = tf.nn.relu(l2_vlt)

	W2_sf = tf.Variable(tf.truncated_normal([hidden_dim[0][1], hidden_dim[1][1]], stddev=0.001), name='Weight_2_sf')
	b2_sf = tf.Variable(tf.zeros([hidden_dim[1][1]]), name='Bias_2_sf')
	l2_sf = tf.add(tf.matmul(l1_sf, W2_sf), b2_sf)
	l2_sf = tf.nn.relu(l2_sf)


with tf.variable_scope('Layer_3'):
	W3 = tf.Variable(tf.truncated_normal([hidden_dim[1][0]+hidden_dim[1][1], hidden_dim[2]], stddev=0.001), name='Weight_3')
	b3 = tf.Variable(tf.zeros([hidden_dim[2]]), name='Bias_3')
	l3 = tf.add(tf.matmul(tf.concat([l2_vlt, l2_sf], axis=1), W3), b3)
	l3 = tf.nn.relu(l3)

with tf.variable_scope('Layer_4'):
	W4 = tf.Variable(tf.truncated_normal([hidden_dim[2]+oth_dim, hidden_dim[3]], stddev=0.001), name='Weight_4')
	b4 = tf.Variable(tf.zeros([hidden_dim[3]]), name='Bias_4')
	l4 = tf.add(tf.matmul(tf.concat([l3, x[:, vlt_dim+sf_dim:vlt_dim+sf_dim+oth_dim]], axis=1), W4), b4)
	l4 = tf.nn.relu(l4)

with tf.variable_scope('Layer_final'):
	W5 = tf.Variable(tf.truncated_normal([hidden_dim[3],1], stddev=0.001), name='Weight_5')
	b5 = tf.Variable(tf.zeros([1]), name='Bias_5')
	output = tf.add(tf.matmul(l4, W5), b5)
	error = output - y
	loss = tf.reduce_mean(tf.square(tf.maximum(q*error, (q-1)*error)) )

with tf.name_scope('Optimizer'):
	train_step = tf.train.AdamOptimizer().minimize(loss)

epochs = 10
batch_size = 64
init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(epochs):
#         # Split data to batches
#         for idx in range(0, X.shape[0], batch_size):
#             batch_data = X[idx : min(idx + batch_size, X.shape[0]),:]
#             batch_labels = labels[idx : min(idx + batch_size, labels.shape[0]),:]
#             feed_dict = {x: batch_data, y: batch_labels}
#             _, c_loss = sess.run([train_step, loss], feed_dict)
#             print(c_loss)