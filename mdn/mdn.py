import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import socket
import sys
if socket.gethostname()=='vision-gpu-1' or socket.gethostname()=='vision-gpu-2':
	sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_r1/lib/python2.7/site-packages/')
else:
	sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_r1_pascal/lib/python2.7/site-packages/')

import tensorflow as tf
import numpy as np
from data_loaders.zhangfeats_loader import zhangfeats_loader 
from arch.layer_factory import layer_factory

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer("feats_height", 28, "")
flags.DEFINE_integer("feats_width", 28, "")
flags.DEFINE_integer("feats_nch", 512, "")
flags.DEFINE_integer("nmix", 8, "GMM components")
flags.DEFINE_boolean("is_train", True, "")
flags.DEFINE_integer("max_epoch", 5, "Max epoch")
flags.DEFINE_float("lr", 1e-5, "Learning rate")

#Dynamical assigned params
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("hidden_size", 64, "VAE latent variable dimension")
flags.DEFINE_integer("updates_per_epoch", 0, "Number of updates per epoch")

def cnn_feedforward(lf, input_tensor, bn_is_training, keep_prob, reuse=False):

	nout = (FLAGS.hidden_size+1)*FLAGS.nmix

	if(reuse == False):
		W_conv1 = lf.weight_variable(name='W_conv1', shape=[5, 5, FLAGS.feats_nch, 384])
		W_conv1_1 = lf.weight_variable(name='W_conv1_1', shape=[5, 5, 384, 320])
		W_conv1_2 = lf.weight_variable(name='W_conv1_2', shape=[5, 5, 320, 288])
		W_conv1_3 = lf.weight_variable(name='W_conv1_3', shape=[5, 5, 288, 256])

		W_conv2 = lf.weight_variable(name='W_conv2', shape=[5, 5, 256, 128])
		W_fc1 = lf.weight_variable(name='W_fc1', shape=[14*14*128, 4096])
		W_fc2 = lf.weight_variable(name='W_fc2', shape=[4096, nout])


		b_fc1 = lf.bias_variable(name='b_fc1', shape=[4096])
		b_fc2 = lf.bias_variable(name='b_fc2', shape=[nout])

	else:
		W_conv1 = lf.weight_variable(name='W_conv1')
		W_conv1_1 = lf.weight_variable(name='W_conv1_1')
		W_conv1_2 = lf.weight_variable(name='W_conv1_2')
		W_conv1_3 = lf.weight_variable(name='W_conv1_3')

		W_conv2 = lf.weight_variable(name='W_conv2')
		W_fc1 = lf.weight_variable(name='W_fc1')
		W_fc2 = lf.weight_variable(name='W_fc2')

		b_fc1 = lf.bias_variable(name='b_fc1')
		b_fc2 = lf.bias_variable(name='b_fc2')

	conv1 = tf.nn.relu(lf.conv2d(input_tensor, W_conv1, stride=1))
	conv1_norm = lf.batch_norm_aiuiuc_wrapper(conv1, bn_is_training, \
			'BN1', reuse_vars=reuse)

	conv1_1 = tf.nn.relu(lf.conv2d(conv1_norm, W_conv1_1, stride=1))
	conv1_1_norm = lf.batch_norm_aiuiuc_wrapper(conv1_1, bn_is_training, \
			'BN1_1', reuse_vars=reuse)
	
	conv1_2 = tf.nn.relu(lf.conv2d(conv1_1_norm, W_conv1_2, stride=1))
	conv1_2_norm = lf.batch_norm_aiuiuc_wrapper(conv1_2, bn_is_training, \
			'BN1_2', reuse_vars=reuse)

	conv1_3 = tf.nn.relu(lf.conv2d(conv1_2_norm, W_conv1_3, stride=2))
	conv1_3_norm = lf.batch_norm_aiuiuc_wrapper(conv1_3, bn_is_training, \
			'BN1_3', reuse_vars=reuse)

	conv2 = tf.nn.relu(lf.conv2d(conv1_3_norm, W_conv2, stride=1))
	conv2_norm = lf.batch_norm_aiuiuc_wrapper(conv2, bn_is_training, \
			'BN2', reuse_vars=reuse)
	
	dropout1 = tf.nn.dropout(conv2_norm, keep_prob)
	flatten1 = tf.reshape(dropout1, [-1, 14*14*128])
	fc1 = tf.tanh(tf.matmul(flatten1, W_fc1)+b_fc1)

	dropout2 = tf.nn.dropout(fc1, keep_prob)
	fc2 = tf.matmul(dropout2, W_fc2)+b_fc2

	return fc2

def get_mixture_coeff(out_fc):
	out_mu = out_fc[..., :FLAGS.hidden_size*FLAGS.nmix]
	out_pi = tf.nn.softmax(out_fc[..., FLAGS.hidden_size*FLAGS.nmix:])
	out_sigma = tf.constant(.1, shape=[FLAGS.batch_size, FLAGS.nmix])
	return out_pi, out_mu, out_sigma

def compute_gmm_loss(gt_tensor, op_tensor_activ, summ=False):
	
	#Replicate ground-truth tensor per mixture component
	gt_tensor_flat = tf.tile(gt_tensor, [FLAGS.nmix, 1])
	
	#Pi, mu, sigma
	op_tensor_pi, op_tensor_mu, op_tensor_sigma = get_mixture_coeff(op_tensor_activ)

	#Flatten means, sigma, pi aligned to gt above
	op_tensor_mu_flat = tf.reshape(op_tensor_mu, [FLAGS.nmix*FLAGS.batch_size, FLAGS.hidden_size])
	op_tensor_sigma_flat = tf.reshape(op_tensor_sigma, [FLAGS.nmix*FLAGS.batch_size])

	#N(t|x, mu, sigma): batch_size x nmix
	op_norm_dist = tf.reshape(tf.div((.5*tf.reduce_sum(tf.square(gt_tensor_flat-op_tensor_mu_flat), \
			reduction_indices=1)), op_tensor_sigma_flat), [FLAGS.batch_size, FLAGS.nmix])
	op_norm_dist_min = tf.reduce_min(op_norm_dist, reduction_indices=1)
	op_norm_dist_minind = tf.to_int32(tf.argmin(op_norm_dist, 1))
	op_pi_minind_flattened = tf.range(0, FLAGS.batch_size)*FLAGS.nmix + op_norm_dist_minind
	op_pi_min = tf.gather(tf.reshape(op_tensor_pi, [-1]), op_pi_minind_flattened)

	if(summ == True):
		gmm_loss = tf.reduce_mean(-tf.log(op_pi_min+1e-30) + op_norm_dist_min, reduction_indices=0)
	else:
		gmm_loss = tf.reduce_mean(op_norm_dist_min, reduction_indices=0)
	
	if(summ == True):
		tf.summary.scalar('gmm_loss', gmm_loss)
		tf.summary.scalar('op_norm_dist_min', tf.reduce_min(op_norm_dist))
		tf.summary.scalar('op_norm_dist_max', tf.reduce_max(op_norm_dist))
		tf.summary.scalar('op_pi_min', tf.reduce_mean(op_pi_min))

	return gmm_loss, op_tensor_pi, op_tensor_mu, op_tensor_sigma 

def optimize(loss, lr):
	optimizer = tf.train.GradientDescentOptimizer(lr)
	return optimizer.minimize(loss)

def save_chkpt(saver, epoch, sess, chkptdir, prefix='model'):
	if not os.path.exists(chkptdir):
		os.makedirs(chkptdir)
	save_path = saver.save(sess, "%s/%s_%06d.ckpt" % (chkptdir, prefix, epoch))
	print("[DEBUG] ############ Model saved in file: %s ################" % save_path)

def load_chkpt(saver, sess, chkptdir):
	ckpt = tf.train.get_checkpoint_state(chkptdir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_fn = ckpt.model_checkpoint_path.replace('//', '/') 
		print('[DEBUG] Loading checkpoint from %s' % ckpt_fn)
		saver.restore(sess, ckpt_fn)
	else:
		raise NameError('[ERROR] No checkpoint found at: %s' % chkptdir)

def main():

	if(len(sys.argv) == 1):
		raise NameError('[ERROR] No dataset key')
	if(sys.argv[1] == 'imagenetval'):
		FLAGS.updates_per_epoch = 49000
		FLAGS.num_test_batches = 1000
		FLAGS.in_featdir = 'data/featslist/imagenetval/' 
		FLAGS.in_lvdir = '/data/ardeshp2/output_imagenetval/' 
	elif(sys.argv[1] == 'lfw'):
		FLAGS.updates_per_epoch = 12233
		FLAGS.num_test_batches = 1000
		FLAGS.in_featdir = 'data/featslist/lfw/' 
		FLAGS.in_lvdir = '/data/ardeshp2/output_lfw/' 
	elif(sys.argv[1] == 'church'):
		FLAGS.updates_per_epoch = 125227
		FLAGS.num_test_batches = 1000
		FLAGS.in_featdir = 'data/featslist/church/' 
		FLAGS.in_lvdir = '/data/ardeshp2/output_church/' 
	else:
		raise NameError('[ERROR] Incorrect dataset key')

	data_loader = zhangfeats_loader(os.path.join(FLAGS.in_featdir, 'list.train.txt'), \
		os.path.join(FLAGS.in_featdir, 'list.test.txt'),\
		os.path.join(FLAGS.in_lvdir, 'lv_color_train.mat.npy'),\
		os.path.join(FLAGS.in_lvdir, 'lv_color_test.mat.npy'))

	#Inputs
	lf = layer_factory()
	input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.feats_height, \
			FLAGS.feats_width, FLAGS.feats_nch])
	output_gt_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.hidden_size])
	is_training = tf.placeholder(tf.bool)
	keep_prob = tf.placeholder(tf.float32)

	#Inference
	with tf.variable_scope('Inference', reuse=False):
		output_activ = cnn_feedforward(lf, input_tensor, is_training, keep_prob, reuse=False)
	
	with tf.variable_scope('Inference', reuse=True):
		output_test_activ = cnn_feedforward(lf, input_tensor, is_training, keep_prob, reuse=True)

	#Loss and gradient descent step
	loss, _, _, _ = compute_gmm_loss(output_gt_tensor, output_activ, summ=True)
	loss_test, pi_test, mu_test, sigma_test = compute_gmm_loss(output_gt_tensor, output_test_activ)

	train_step = optimize(loss, FLAGS.lr)

	#Standard steps		
	check_nan_op = tf.add_check_numerics_ops()
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=0)
	summary_op = tf.summary.merge_all()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	train_writer = tf.summary.FileWriter(os.path.join(FLAGS.in_lvdir, 'logs_mdn'), sess.graph)

	sess.run(init)
	
	if(FLAGS.is_train):			
		for epoch  in range(FLAGS.max_epoch):
			training_loss = 0.

			data_loader.random_reset()
			for i in range(FLAGS.updates_per_epoch):
				batch, batch_gt = data_loader.train_next_batch(FLAGS.batch_size)
				feed_dict = {input_tensor:batch, output_gt_tensor:batch_gt, \
					is_training:True, keep_prob:.75}
				_, _, loss_value, summary_str = sess.run(\
					[check_nan_op, train_step, loss, summary_op], \
					feed_dict)
				train_writer.add_summary(summary_str, epoch*FLAGS.updates_per_epoch+i)
				training_loss = training_loss + loss_value

			print('[DEBUG] Epoch# %d, Loss: %f' % (epoch, \
			(training_loss*1.)/FLAGS.updates_per_epoch))

			save_chkpt(saver, epoch, sess, os.path.join(FLAGS.in_lvdir, 'models_mdn'), \
        prefix='model_%d_exp' % FLAGS.nmix)
	else:
			load_chkpt(saver, sess, os.path.join(FLAGS.in_lvdir, 'models_mdn'))

	test_loss = 0.
	data_loader.reset()
	lv_test_codes = np.zeros((0, (FLAGS.hidden_size+1+1)*FLAGS.nmix), dtype='f')
	for i in range(FLAGS.num_test_batches):
		batch, batch_gt = data_loader.test_next_batch(FLAGS.batch_size)
		feed_dict = {input_tensor:batch, output_gt_tensor:batch_gt, \
			is_training:False, keep_prob:1.}
		_, loss_value, output_pi, output_mu, output_sigma = \
			sess.run([check_nan_op, loss_test, pi_test, mu_test, sigma_test], feed_dict)

		test_loss = test_loss + loss_value
		output = np.concatenate((output_mu, output_sigma, output_pi), axis=1)
		lv_test_codes = np.concatenate((lv_test_codes, output), axis=0)
	
	print('[DEBUG] Test Loss: %f' % ((test_loss*1.)/FLAGS.num_test_batches))
	np.save(os.path.join(FLAGS.in_lvdir, 'lv_color_mdn_test.mat'), lv_test_codes)		
	print(lv_test_codes.shape)
	
	sess.close()

if __name__ == "__main__":
	main()
