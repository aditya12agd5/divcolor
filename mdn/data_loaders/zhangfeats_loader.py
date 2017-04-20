import cv2
import glob
import math
import numpy as np

class zhangfeats_loader:

	def __init__(self, list_train_fn, list_test_fn, lv_train_fn, lv_test_fn, \
			featshape=(512, 28, 28)):
		
		self.train_img_fns = [] 
		self.test_img_fns = []
 
		with open(list_train_fn, 'r') as ftr:
			for img_fn in ftr:
				self.train_img_fns.append(img_fn.strip('\n'))
	
		with open(list_test_fn, 'r') as fte:
			for img_fn in fte:
				self.test_img_fns.append(img_fn.strip('\n'))
	
		self.lv_train = np.load(lv_train_fn)
		self.lv_test = np.load(lv_test_fn)
		self.hidden_size = np.int_((self.lv_train.shape[1]*1.)/2.)
		
		self.train_img_fns = self.train_img_fns[:self.lv_train.shape[0]]
		self.test_img_fns = self.test_img_fns[:self.lv_test.shape[0]]
		self.featshape = featshape

		self.train_img_num = len(self.train_img_fns)
		self.test_img_num = len(self.test_img_fns)
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = range(self.train_img_num)
		self.test_shuff_ids = range(self.test_img_num)
		
	def reset(self):
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = range(self.train_img_num)
		self.test_shuff_ids = range(self.test_img_num)
	
	def random_reset(self):
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = np.random.permutation(self.train_img_num)
		self.test_shuff_ids = range(self.test_img_num)

	def train_next_batch(self, batch_size):
		batch = np.zeros((batch_size, self.featshape[2], \
				self.featshape[1], self.featshape[0]), dtype='f')
		batch_gt = np.zeros((batch_size, self.hidden_size), dtype='f')

		if(self.train_batch_head + batch_size >= self.train_img_num):
			self.train_shuff_ids = np.random.permutation(self.train_img_num)
			self.train_batch_head = 0
		
		for i_n, i in enumerate(range(self.train_batch_head, self.train_batch_head+batch_size)):
			currid = self.train_shuff_ids[i]
			featobj = np.load(self.train_img_fns[currid])
			feats = featobj['arr_0'].reshape(self.featshape[0], self.featshape[1], \
					self.featshape[2])
			feats2d = feats.reshape(self.featshape[0], -1).T
			feats3d = feats2d.reshape(self.featshape[1], self.featshape[2], \
					self.featshape[0])
			batch[i_n, ...] = feats3d
			eps = np.random.normal(loc=0., scale=1., size=(self.hidden_size))
			batch_gt[i_n, ...] = self.lv_train[currid, :self.hidden_size] \
				+ eps*self.lv_train[currid, self.hidden_size:]

		self.train_batch_head = self.train_batch_head + batch_size

		return batch, batch_gt

	def test_next_batch(self, batch_size):
		batch = np.zeros((batch_size, self.featshape[2], \
				self.featshape[1], self.featshape[0]), dtype='f')
		batch_gt = np.zeros((batch_size, self.hidden_size), dtype='f')

		if(self.test_batch_head + batch_size > self.test_img_num):
			self.test_shuff_ids = range(self.test_img_num) 
			self.test_batch_head = 0
		
		for i_n, i in enumerate(range(self.test_batch_head, self.test_batch_head+batch_size)):
			currid = self.test_shuff_ids[i]
			featobj = np.load(self.test_img_fns[currid])
			feats = featobj['arr_0'].reshape(self.featshape[0], self.featshape[1], \
					self.featshape[2])
			feats2d = feats.reshape(self.featshape[0], -1).T
			feats3d = feats2d.reshape(self.featshape[1], self.featshape[2], \
					self.featshape[0])
			batch[i_n, ...] = feats3d
			eps = np.random.normal(loc=0., scale=1., size=(self.hidden_size))
			batch_gt[i_n, ...] = self.lv_test[currid, :self.hidden_size] \
				+eps*self.lv_test[currid, self.hidden_size:]

		self.test_batch_head = self.test_batch_head + batch_size
		return batch, batch_gt
