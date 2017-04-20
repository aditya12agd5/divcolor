import os
import numpy as np
import tensorflow as tf
from layer_factory import layer_factory
from tensorflow.python.framework import tensor_shape

class vae_wo_skipconn:

	def __init__(self, flags, nch=2, condinference_flag=False):
		self.flags = flags
		self.nch = nch
		self.layer_factory = layer_factory()
		self.condinference_flag = condinference_flag
	
	#Returns handles to input placeholders
	def inputs(self):
		inp_img = tf.placeholder(tf.float32, [self.flags.batch_size, \
			self.nch * self.flags.img_height * self.flags.img_width])
		inp_greylevel = tf.placeholder(tf.float32, [self.flags.batch_size, \
			self.flags.img_height * self.flags.img_width])
		inp_latent = tf.placeholder(tf.float32, [self.flags.batch_size, \
			self.flags.hidden_size])
		is_training = tf.placeholder(tf.bool)
		is_training_dec = tf.placeholder(tf.bool)
		keep_prob = tf.placeholder(tf.float32)
		kl_weight = tf.placeholder(tf.float32)
		lossweights = tf.placeholder(tf.float32, [self.flags.batch_size, \
			self.nch * self.flags.img_height * self.flags.img_width])

		return inp_img, inp_greylevel, inp_latent, is_training, keep_prob, kl_weight, lossweights

	#Takes input placeholders, builds inference graph and returns net. outputs
	def inference(self, inp_img, inp_greylevel, inp_latent, is_training, keep_prob):

		with tf.variable_scope('Inference', reuse=False) as sc:
			z1_train = self.__encoder(sc, inp_img, is_training, keep_prob, \
					in_nch=self.nch, reuse=False)
			epsilon_train = tf.truncated_normal([self.flags.batch_size, self.flags.hidden_size])
			mean_train = z1_train[:, :self.flags.hidden_size]
			stddev_train = tf.sqrt(tf.exp(z1_train[:, self.flags.hidden_size:]))
			z1_sample = mean_train + epsilon_train * stddev_train
			output_train = self.__decoder(sc, is_training, inp_greylevel, z1_sample, reuse=False)

		with tf.variable_scope('Inference', reuse=True) as sc:
			if(self.condinference_flag == False):
				z1_test = self.__encoder(sc, inp_img, is_training, keep_prob, \
					in_nch=self.nch, reuse=True)
				epsilon_test = tf.truncated_normal([self.flags.batch_size, self.flags.hidden_size])
				mean_test = z1_test[:, :self.flags.hidden_size]
				stddev_test = tf.sqrt(tf.exp(z1_test[:, self.flags.hidden_size:]))
				z1_sample = mean_test + epsilon_test * stddev_test
				tf.stop_gradient(z1_sample) #Fix the encoder
				output_test = self.__decoder(sc, is_training, inp_greylevel, z1_sample, reuse=True)
				output_condinference = None
			else:
				mean_test = None
				stddev_test = None
				output_test = None
				output_condinference = self.__decoder(sc, is_training, inp_greylevel, inp_latent,\
					reuse=True)

		return mean_train, stddev_train, output_train, mean_test, stddev_test, \
			output_test, output_condinference
	
	#Takes net. outputs and computes loss for vae(enc+dec)
	def loss(self, target_tensor, op_tensor, mean, stddev, kl_weight, lossweights, epsilon=1e-6, \
    is_regression=True):
		
		kl_loss = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) \
					- tf.log(tf.maximum(tf.square(stddev), epsilon)) - 1.0))

		recon_loss_chi = tf.reduce_mean(tf.sqrt(tf.reduce_sum( \
			     lossweights*tf.square(target_tensor-op_tensor), 1)), 0)

    #Load Principle components
		np_pcvec = np.transpose(np.load(os.path.join(self.flags.pc_dir, 'components.mat.npy')))
		np_pcvar = 1./np.load(os.path.join(self.flags.pc_dir, 'exp_variance.mat.npy'))
		np_pcvec = np_pcvec[:, :self.flags.pc_comp]
		np_pcvar = np_pcvar[:self.flags.pc_comp]
		pcvec = tf.constant(np_pcvec)
		pcvar = tf.constant(np_pcvar)

		projmat_op = tf.matmul(op_tensor, pcvec)
		projmat_target = tf.matmul(target_tensor, pcvec)
		weightmat = tf.tile(tf.reshape(pcvar, [1, self.flags.pc_comp]), [self.flags.batch_size, 1])
		loss_topk_pc = tf.reduce_mean(tf.reduce_sum(\
      tf.multiply(tf.square(projmat_op-projmat_target), weightmat), 1), 0)

		res_op = op_tensor
		res_target = target_tensor
		for npc in range(self.flags.pc_comp):
			pcvec_curr = tf.tile(tf.reshape(tf.transpose(pcvec[:, npc]), [1, -1]), \
				[self.flags.batch_size, 1])
			projop_curr = tf.tile(tf.reshape(projmat_op[:, npc], [self.flags.batch_size, 1]), \
				[1, self.nch * self.flags.img_height * self.flags.img_width])

			projtarget_curr = tf.tile(tf.reshape(projmat_target[:, npc], [self.flags.batch_size, 1]), \
				[1, self.nch * self.flags.img_height * self.flags.img_width])
			
			res_op = tf.subtract(res_op, tf.multiply(projop_curr, pcvec_curr))
			res_target = tf.subtract(res_target, tf.multiply(projtarget_curr, pcvec_curr))

		res_error = tf.reduce_sum(tf.square(res_op-res_target), 1)
		res_error_weight = tf.tile(tf.reshape(pcvar[self.flags.pc_comp-1], [1, 1]), [self.flags.batch_size, 1])
		loss_res_pc = tf.reduce_mean(tf.multiply(\
      tf.reshape(res_error, [self.flags.batch_size, 1]), res_error_weight))

		recon_loss = recon_loss_chi + (1e-1)*(loss_topk_pc + loss_res_pc)

		if(self.nch == 2):
			target_tensor2d = tf.reshape(target_tensor, [self.flags.batch_size, \
				self.flags.img_height, self.flags.img_width, self.nch])	
			op_tensor2d = tf.reshape(op_tensor, [self.flags.batch_size, \
				self.flags.img_height, self.flags.img_width, self.nch])
			[n,w,h,c] = target_tensor2d.get_shape().as_list()
			dv = tf.square((target_tensor2d[:,1:,:h-1,:] - target_tensor2d[:,:w-1,:h-1,:])
				- (op_tensor2d[:,1:,:h-1,:] - op_tensor2d[:,:w-1,:h-1,:]))
			dh = tf.square((target_tensor2d[:,:w-1,1:,:] - target_tensor2d[:,:w-1,:h-1,:])
				- (op_tensor2d[:,:w-1,1:,:] - op_tensor2d[:,:w-1,:h-1,:]))
			grad_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(dv+dh,[1,2,3])))
			recon_loss = recon_loss + (1e-3)*grad_loss


		loss = kl_weight*kl_loss + recon_loss

		tf.summary.scalar('grad_loss', grad_loss)
		tf.summary.scalar('kl_loss', kl_loss)
		tf.summary.scalar('recon_loss_chi', recon_loss_chi)
		tf.summary.scalar('recon_loss', recon_loss)
		return loss
	
	#Takes loss and returns GD train step
	def optimize(self, loss, epsilon):
		train_step = tf.train.AdamOptimizer(self.flags.lr_vae, epsilon=epsilon).minimize(loss)
		return train_step
	
	def __encoder(self, scope, input_tensor, bn_is_training, keep_prob, in_nch=1, reuse=False):
	
		lf = self.layer_factory
		
		input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
				self.flags.img_height, self.flags.img_width, in_nch])

		if(self.nch == 1 and reuse==False):
			tf.image_summary('summ_input_tensor2d', input_tensor2d, max_images=10)

		nch = tensor_shape.as_dimension(input_tensor2d.get_shape()[3]).value

		if(reuse==False):
			W_conv1 = lf.weight_variable(name='W_conv1', shape=[5, 5, nch, 128])
			W_conv2 = lf.weight_variable(name='W_conv2', shape=[5, 5, 128, 256])
			W_conv3 = lf.weight_variable(name='W_conv3', shape=[5, 5, 256, 512])
			W_conv4 = lf.weight_variable(name='W_conv4', shape=[4, 4, 512, 1024])
			W_fc1 = lf.weight_variable(name='W_fc1', shape=[4*4*1024, self.flags.hidden_size * 2])

			b_conv1 = lf.bias_variable(name='b_conv1', shape=[128])
			b_conv2 = lf.bias_variable(name='b_conv2', shape=[256])
			b_conv3 = lf.bias_variable(name='b_conv3', shape=[512])
			b_conv4 = lf.bias_variable(name='b_conv4', shape=[1024])
			b_fc1 = lf.bias_variable(name='b_fc1', shape=[self.flags.hidden_size * 2])
		else:
			W_conv1 = lf.weight_variable(name='W_conv1')
			W_conv2 = lf.weight_variable(name='W_conv2')
			W_conv3 = lf.weight_variable(name='W_conv3')
			W_conv4 = lf.weight_variable(name='W_conv4')
			W_fc1 = lf.weight_variable(name='W_fc1')

			b_conv1 = lf.bias_variable(name='b_conv1')
			b_conv2 = lf.bias_variable(name='b_conv2')
			b_conv3 = lf.bias_variable(name='b_conv3')
			b_conv4 = lf.bias_variable(name='b_conv4')
			b_fc1 = lf.bias_variable(name='b_fc1')
		
		conv1 = tf.nn.relu(lf.conv2d(input_tensor2d, W_conv1, stride=2) + b_conv1)
		conv1_norm = lf.batch_norm_aiuiuc_wrapper(conv1, bn_is_training, \
			'BN1', reuse_vars=reuse)

		conv2 = tf.nn.relu(lf.conv2d(conv1_norm, W_conv2, stride=2) + b_conv2)
		conv2_norm = lf.batch_norm_aiuiuc_wrapper(conv2, bn_is_training, \
			'BN2', reuse_vars=reuse)

		conv3 = tf.nn.relu(lf.conv2d(conv2_norm, W_conv3, stride=2) + b_conv3)
		conv3_norm = lf.batch_norm_aiuiuc_wrapper(conv3, bn_is_training, \
			'BN3', reuse_vars=reuse)

		conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=2) + b_conv4)
		conv4_norm = lf.batch_norm_aiuiuc_wrapper(conv4, bn_is_training, \
			'BN4', reuse_vars=reuse)
		
		dropout1 = tf.nn.dropout(conv4_norm, keep_prob)
		flatten1 = tf.reshape(dropout1, [-1, 4*4*1024])

		fc1 = tf.matmul(flatten1, W_fc1)+b_fc1

		return fc1

	def __decoder(self, scope, bn_is_training, inp_greylevel, z1_sample, reuse=False):

		lf = self.layer_factory

		if(reuse == False):
			W_deconv1 = lf.weight_variable(name='W_deconv1', shape=[4, 4, self.flags.hidden_size, 1024])
			W_deconv2 = lf.weight_variable(name='W_deconv2', shape=[5, 5, 1024, 512])
			W_deconv3 = lf.weight_variable(name='W_deconv3', shape=[5, 5, 514, 256])
			W_deconv4 = lf.weight_variable(name='W_deconv4', shape=[5, 5, 258, 128])
			W_deconv5 = lf.weight_variable(name='W_deconv5', shape=[5, 5, 128, self.nch])
		
			b_deconv1 = lf.bias_variable(name='b_deconv1', shape=[1024])
			b_deconv2 = lf.bias_variable(name='b_deconv2', shape=[512])
			b_deconv3 = lf.bias_variable(name='b_deconv3', shape=[256])
			b_deconv4 = lf.bias_variable(name='b_deconv4', shape=[128])
			b_deconv5 = lf.bias_variable(name='b_deconv5', shape=[self.nch])
		else:
			W_deconv1 = lf.weight_variable(name='W_deconv1')
			W_deconv2 = lf.weight_variable(name='W_deconv2')
			W_deconv3 = lf.weight_variable(name='W_deconv3')
			W_deconv4 = lf.weight_variable(name='W_deconv4')
			W_deconv5 = lf.weight_variable(name='W_deconv5')
		
			b_deconv1 = lf.bias_variable(name='b_deconv1')
			b_deconv2 = lf.bias_variable(name='b_deconv2')
			b_deconv3 = lf.bias_variable(name='b_deconv3')
			b_deconv4 = lf.bias_variable(name='b_deconv4')
			b_deconv5 = lf.bias_variable(name='b_deconv5')

		inp_greylevel2d = tf.reshape(inp_greylevel, [self.flags.batch_size, \
			self.flags.img_height, self.flags.img_width, 1])
		input_concat2d = tf.reshape(z1_sample, [self.flags.batch_size, 1, 1, self.flags.hidden_size])

		deconv1_upsamp = tf.image.resize_images(input_concat2d, [4, 4])
		deconv1 = tf.nn.relu(lf.conv2d(deconv1_upsamp, W_deconv1, stride=1) + b_deconv1)
		deconv1_norm = lf.batch_norm_aiuiuc_wrapper(deconv1, bn_is_training, \
			'BN_deconv1', reuse_vars=reuse)
		
		deconv2_upsamp = tf.image.resize_images(deconv1_norm, [8, 8])
		deconv2 = tf.nn.relu(lf.conv2d(deconv2_upsamp, W_deconv2, stride=1) + b_deconv2)
		deconv2_norm = lf.batch_norm_aiuiuc_wrapper(deconv2, bn_is_training, \
			'BN_deconv2', reuse_vars=reuse)

		deconv3_upsamp = tf.image.resize_images(deconv2_norm, [16, 16])
		grey_deconv3_dv, grey_deconv3_dh = self.__get_gradients(inp_greylevel2d, \
			shape=[16, 16]) 
		deconv3_upsamp_edge = tf.concat([deconv3_upsamp, grey_deconv3_dv, grey_deconv3_dh], 3)
		deconv3 = tf.nn.relu(lf.conv2d(deconv3_upsamp_edge, W_deconv3, stride=1) + b_deconv3)
		deconv3_norm = lf.batch_norm_aiuiuc_wrapper(deconv3, bn_is_training, \
			'BN_deconv3', reuse_vars=reuse)

		deconv4_upsamp = tf.image.resize_images(deconv3_norm, [32, 32])
		grey_deconv4_dv, grey_deconv4_dh = self.__get_gradients(inp_greylevel2d, \
			shape=[32, 32]) 
		deconv4_upsamp_edge = tf.concat([deconv4_upsamp, grey_deconv4_dv, grey_deconv4_dh], 3)
		deconv4 = tf.nn.relu(lf.conv2d(deconv4_upsamp_edge, W_deconv4, stride=1) + b_deconv4)
		deconv4_norm = lf.batch_norm_aiuiuc_wrapper(deconv4, bn_is_training, \
			'BN_deconv4', reuse_vars=reuse)

		deconv5_upsamp = tf.image.resize_images(deconv4_norm, [64, 64])
		deconv5 = lf.conv2d(deconv5_upsamp, W_deconv5, stride=1) + b_deconv5
		deconv5_norm = lf.batch_norm_aiuiuc_wrapper(deconv5, bn_is_training, \
			'BN_deconv5', reuse_vars=reuse)
		
		decoded_ch = tf.reshape(tf.tanh(deconv5_norm), \
			[self.flags.batch_size, self.flags.img_height*self.flags.img_width*self.nch])

		return decoded_ch

	def __get_gradients(self, in_tensor2d, shape=None):
		if(shape is not None):
			in_tensor = tf.image.resize_images(in_tensor2d, [shape[0], shape[1]])	
		else:
			in_tensor = in_tensor2d
		[n,w,h,c] = in_tensor.get_shape().as_list()
		dvert = in_tensor[:,1:,:h,:] - in_tensor[:,:w-1,:h,:]
		dvert_padded = tf.concat([tf.constant(0., shape=[n, 1, h, c]), dvert], 1)
		dhorz = in_tensor[:,:w,1:,:] - in_tensor[:,:w,:h-1,:]
		dhorz_padded = tf.concat([tf.constant(0., shape=[n, w, 1, c]), dhorz], 2)
		return dvert_padded, dhorz_padded
