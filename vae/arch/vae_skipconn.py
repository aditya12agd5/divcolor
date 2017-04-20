import tensorflow as tf
from layer_factory import layer_factory
from tensorflow.python.framework import tensor_shape

class vae_skipconn:

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
		keep_prob = tf.placeholder(tf.float32)
		kl_weight = tf.placeholder(tf.float32)
		lossweights = tf.placeholder(tf.float32, [self.flags.batch_size, \
			self.nch * self.flags.img_height * self.flags.img_width])
		return inp_img, inp_greylevel, inp_latent, is_training, keep_prob, kl_weight, lossweights

	#Takes input placeholders, builds inference graph and returns net. outputs
	def inference(self, inp_img, inp_greylevel, inp_latent, is_training, keep_prob):

		with tf.variable_scope('Inference', reuse=False) as sc:
			gfeat32, gfeat16, gfeat8, gfeat4 = \
				self.__cond_encoder(sc, inp_greylevel, is_training, keep_prob, \
					in_nch=1, reuse=False)
			z1_train = self.__encoder(sc, inp_img, is_training, keep_prob, \
					in_nch=self.nch, reuse=False)
			epsilon_train = tf.truncated_normal([self.flags.batch_size, self.flags.hidden_size])
			mean_train = z1_train[:, :self.flags.hidden_size]
			stddev_train = tf.sqrt(tf.exp(z1_train[:, self.flags.hidden_size:]))
			z1_sample = mean_train + epsilon_train * stddev_train
			output_train = self.__decoder(sc, gfeat32, gfeat16, gfeat8, gfeat4, \
					is_training, inp_greylevel, z1_sample, reuse=False)

		with tf.variable_scope('Inference', reuse=True) as sc:
			gfeat32, gfeat16, gfeat8, gfeat4 = \
				self.__cond_encoder(sc, inp_greylevel, is_training, keep_prob, \
					in_nch=1, reuse=True)

			if(self.condinference_flag == False):
				z1_test = self.__encoder(sc, inp_img, is_training, keep_prob, \
					in_nch=self.nch, reuse=True)
				epsilon_test = tf.truncated_normal([self.flags.batch_size, self.flags.hidden_size])
				mean_test = z1_test[:, :self.flags.hidden_size]
				stddev_test = tf.sqrt(tf.exp(z1_test[:, self.flags.hidden_size:]))
				z1_sample = mean_test + epsilon_test * stddev_test
				tf.stop_gradient(z1_sample) #Fix the encoder
				output_test = self.__decoder(sc, gfeat32, gfeat16, gfeat8, gfeat4, \
					is_training, inp_greylevel, z1_sample, reuse=True)
				output_condinference = None
			else:
				mean_test = None
				stddev_test = None
				output_test = None
				output_condinference = self.__decoder(sc, gfeat32, gfeat16, gfeat8,\
					gfeat4, is_training, inp_greylevel, inp_latent, reuse=True)

		return mean_train, stddev_train, output_train, mean_test, stddev_test, \
			output_test, output_condinference
	
	#Takes net. outputs and computes loss for vae(enc+dec)
	def loss(self, target_tensor, op_tensor, mean, stddev, kl_weight, lossweights, epsilon=1e-6):
		
		kl_loss = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) \
					- tf.log(tf.maximum(tf.square(stddev), epsilon)) - 1.0))

		recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum( \
			     lossweights*tf.square(target_tensor-op_tensor), 1)), 0)

		recon_loss_l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum( \
			     tf.square(target_tensor-op_tensor), 1)), 0)

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
		tf.summary.scalar('kl_loss', kl_loss)
		tf.summary.scalar('grad_loss', grad_loss)
		tf.summary.scalar('recon_loss', recon_loss)
		tf.summary.scalar('recon_loss_l2', recon_loss_l2)
		tf.summary.scalar('loss', loss)
		return loss
	
	#Takes loss and returns GD train step
	def optimize(self, loss, epsilon):
	    train_step = tf.train.AdamOptimizer(self.flags.lr_vae, epsilon=epsilon).minimize(loss) 
	    return train_step
	
	def __cond_encoder(self, scope, input_tensor, bn_is_training, keep_prob, in_nch=1, reuse=False):
	
		lf = self.layer_factory
		input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
			self.flags.img_height, self.flags.img_width, 1])
		nch = tensor_shape.as_dimension(input_tensor2d.get_shape()[3]).value
		nout = self.flags.hidden_size
		
		if(reuse == False):
			W_conv1 = lf.weight_variable(name='W_conv1_cond', shape=[5, 5, nch, 128])
			W_conv2 = lf.weight_variable(name='W_conv2_cond', shape=[5, 5, 128, 256])
			W_conv3 = lf.weight_variable(name='W_conv3_cond', shape=[5, 5, 256, 512])
			W_conv4 = lf.weight_variable(name='W_conv4_cond', shape=[4, 4, 512, self.flags.hidden_size])
			
			b_conv1 = lf.bias_variable(name='b_conv1_cond', shape=[128])
			b_conv2 = lf.bias_variable(name='b_conv2_cond', shape=[256])
			b_conv3 = lf.bias_variable(name='b_conv3_cond', shape=[512])
			b_conv4 = lf.bias_variable(name='b_conv4_cond', shape=[self.flags.hidden_size])
		else:
			W_conv1 = lf.weight_variable(name='W_conv1_cond')
			W_conv2 = lf.weight_variable(name='W_conv2_cond')
			W_conv3 = lf.weight_variable(name='W_conv3_cond')
			W_conv4 = lf.weight_variable(name='W_conv4_cond')
			
			b_conv1 = lf.bias_variable(name='b_conv1_cond')
			b_conv2 = lf.bias_variable(name='b_conv2_cond')
			b_conv3 = lf.bias_variable(name='b_conv3_cond')
			b_conv4 = lf.bias_variable(name='b_conv4_cond')

		conv1 = tf.nn.relu(lf.conv2d(input_tensor2d, W_conv1, stride=2) + b_conv1)
		conv1_norm = lf.batch_norm_aiuiuc_wrapper(conv1, bn_is_training, \
				'BN1_cond', reuse_vars=reuse)
	
		conv2 = tf.nn.relu(lf.conv2d(conv1_norm, W_conv2, stride=2) + b_conv2)
		conv2_norm = lf.batch_norm_aiuiuc_wrapper(conv2, bn_is_training, \
				'BN2_cond', reuse_vars=reuse)
	
		conv3 = tf.nn.relu(lf.conv2d(conv2_norm, W_conv3, stride=2) + b_conv3)
		conv3_norm = lf.batch_norm_aiuiuc_wrapper(conv3, bn_is_training, \
				'BN3_cond', reuse_vars=reuse)

		conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=2) + b_conv4)
		conv4_norm = lf.batch_norm_aiuiuc_wrapper(conv4, bn_is_training, \
				'BN4_cond', reuse_vars=reuse)

		return conv1_norm, conv2_norm, conv3_norm, conv4_norm

	def __encoder(self, scope, input_tensor, bn_is_training, keep_prob, in_nch=2, reuse=False):
	
		lf = self.layer_factory
		
		input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
				self.flags.img_height, self.flags.img_width, in_nch])

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

	def __decoder(self, scope, gfeat32, gfeat16, gfeat8, gfeat4, bn_is_training, inp_greylevel,\
			 z1_sample, reuse=False):

		lf = self.layer_factory

		if(reuse == False):
			W_deconv1 = lf.weight_variable(name='W_deconv1', shape=[4, 4, self.flags.hidden_size, 1024])
			W_deconv2 = lf.weight_variable(name='W_deconv2', shape=[5, 5, 1024+512, 512])
			W_deconv3 = lf.weight_variable(name='W_deconv3', shape=[5, 5, 512+256, 256])
			W_deconv4 = lf.weight_variable(name='W_deconv4', shape=[5, 5, 256+128, 128])
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
		input2d = tf.reshape(z1_sample, [self.flags.batch_size, 1, 1, self.flags.hidden_size])
		deconv1_upsamp = tf.image.resize_images(input2d, [4, 4])

		deconv1_upsamp_sc = tf.multiply(deconv1_upsamp, gfeat4)

		deconv1 = tf.nn.relu(lf.conv2d(deconv1_upsamp_sc, W_deconv1, stride=1) + b_deconv1)
		deconv1_norm = lf.batch_norm_aiuiuc_wrapper(deconv1, bn_is_training, \
			'BN_deconv1', reuse_vars=reuse)
		
		deconv2_upsamp = tf.image.resize_images(deconv1_norm, [8, 8])
		deconv2_upsamp_sc = tf.concat([deconv2_upsamp, gfeat8], 3)
		deconv2 = tf.nn.relu(lf.conv2d(deconv2_upsamp_sc, W_deconv2, stride=1) + b_deconv2)
		deconv2_norm = lf.batch_norm_aiuiuc_wrapper(deconv2, bn_is_training, \
			'BN_deconv2', reuse_vars=reuse)

		deconv3_upsamp = tf.image.resize_images(deconv2_norm, [16, 16])
		deconv3_upsamp_sc = tf.concat([deconv3_upsamp, gfeat16], 3)
		deconv3 = tf.nn.relu(lf.conv2d(deconv3_upsamp_sc, W_deconv3, stride=1) + b_deconv3)
		deconv3_norm = lf.batch_norm_aiuiuc_wrapper(deconv3, bn_is_training, \
			'BN_deconv3', reuse_vars=reuse)

		deconv4_upsamp = tf.image.resize_images(deconv3_norm, [32, 32])
		deconv4_upsamp_sc = tf.concat([deconv4_upsamp, gfeat32], 3)
		deconv4 = tf.nn.relu(lf.conv2d(deconv4_upsamp_sc, W_deconv4, stride=1) + b_deconv4)
		deconv4_norm = lf.batch_norm_aiuiuc_wrapper(deconv4, bn_is_training, \
			'BN_deconv4', reuse_vars=reuse)

		deconv5_upsamp = tf.image.resize_images(deconv4_norm, [64, 64])
		deconv5 = lf.conv2d(deconv5_upsamp, W_deconv5, stride=1) + b_deconv5
		deconv5_norm = lf.batch_norm_aiuiuc_wrapper(deconv5, bn_is_training, \
			'BN_deconv5', reuse_vars=reuse)
		
		decoded_ch = tf.reshape(tf.tanh(deconv5_norm), \
			[self.flags.batch_size, self.flags.img_height*self.flags.img_width*self.nch])

		return decoded_ch
