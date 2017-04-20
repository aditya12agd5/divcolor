import cv2
import glob
import math
import numpy as np
import os

class lab_imageloader:

  def __init__(self, data_directory, out_directory, listdir=None, shape=(64, 64), \
        subdir=False, ext='JPEG', outshape=(256, 256)):

    if(listdir == None):
      if(subdir==False):
        self.img_fns = glob.glob('%s/*.%s' % (data_directory, ext))
      else:
        self.img_fns = glob.glob('%s/*/*.%s' % (data_directory, ext))
      np.random.seed(seed=0)
      selectids = np.random.permutation(len(self.img_fns))
      fact_train = .9
      self.train_img_fns = \
        [self.img_fns[l] for l in selectids[:np.int_(np.round(fact_train*len(self.img_fns)))]]
      self.test_img_fns = \
        [self.img_fns[l] for l in selectids[np.int_(np.round(fact_train*len(self.img_fns))):]]
    else:
      self.train_img_fns = []
      self.test_img_fns = []
      with open('%s/list.train.vae.txt' % listdir, 'r') as ftr:
        for img_fn in ftr:
          self.train_img_fns.append(img_fn.strip('\n'))
      
      with open('%s/list.test.vae.txt' % listdir, 'r') as fte:
        for img_fn in fte:
          self.test_img_fns.append(img_fn.strip('\n'))

    self.train_img_num = len(self.train_img_fns)
    self.test_img_num = len(self.test_img_fns)
    self.train_batch_head = 0
    self.test_batch_head = 0
    self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
    self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
    self.shape = shape
    self.outshape = outshape
    self.out_directory = out_directory
    self.lossweights = None

    countbins = 1./np.load('data/zhang_weights/prior_probs.npy')
    binedges = np.load('data/zhang_weights/ab_quantize.npy').reshape(2, 313)
    lossweights = {}  
    for i in range(313):
      if binedges[0, i] not in lossweights:
        lossweights[binedges[0, i]] = {}
      lossweights[binedges[0,i]][binedges[1,i]] = countbins[i]
    self.binedges = binedges
    self.lossweights = lossweights

  def reset(self):
    self.train_batch_head = 0
    self.test_batch_head = 0
    self.train_shuff_ids = range(len(self.train_img_fns))
    self.test_shuff_ids = range(len(self.test_img_fns))
  
  def random_reset(self):
    self.train_batch_head = 0
    self.test_batch_head = 0
    self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
    self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
  
  def train_next_batch(self, batch_size, nch):
    batch = np.zeros((batch_size, nch*np.prod(self.shape)), dtype='f')
    batch_lossweights = np.ones((batch_size, nch*np.prod(self.shape)), dtype='f')
    batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')
    batch_recon_const_outres = np.zeros((batch_size, np.prod(self.outshape)), dtype='f')

    if(self.train_batch_head + batch_size >= len(self.train_img_fns)):
      self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
      self.train_batch_head = 0

    for i_n, i in enumerate(range(self.train_batch_head, self.train_batch_head+batch_size)):
      currid = self.train_shuff_ids[i]
      img_large = cv2.imread(self.train_img_fns[currid])

      if(self.shape is not None):
        img = cv2.resize(img_large, (self.shape[0], self.shape[1]))
        img_outres = cv2.resize(img_large, (self.outshape[0], self.outshape[1]))

      img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)
      batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
      batch_recon_const_outres[i_n, ...] = ((img_lab_outres[..., 0].reshape(-1)*1.)-128.)/128.
      batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
        ((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)

      if(self.lossweights is not None):
        batch_lossweights[i_n, ...] = self.__get_lossweights(batch[i_n, ...])

    self.train_batch_head = self.train_batch_head + batch_size

    return batch, batch_recon_const, batch_lossweights, batch_recon_const_outres

  def test_next_batch(self, batch_size, nch):
    batch = np.zeros((batch_size, nch*np.prod(self.shape)), dtype='f')
    batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')
    batch_recon_const_outres = np.zeros((batch_size, np.prod(self.outshape)), dtype='f')
    batch_imgnames = []
    if(self.test_batch_head + batch_size > len(self.test_img_fns)):
      self.test_batch_head = 0

    for i_n, i in enumerate(range(self.test_batch_head, self.test_batch_head+batch_size)):
      currid = self.test_shuff_ids[i]
      img_large = cv2.imread(self.test_img_fns[currid])
      batch_imgnames.append(self.test_img_fns[currid].split('/')[-1])
      if(self.shape is not None):
        img = cv2.resize(img_large, (self.shape[1], self.shape[0]))
        img_outres = cv2.resize(img_large, (self.outshape[0], self.outshape[1]))

      img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)

      batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
      batch_recon_const_outres[i_n, ...] = ((img_lab_outres[..., 0].reshape(-1)*1.)-128.)/128.
      batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
        ((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)

    self.test_batch_head = self.test_batch_head + batch_size

    return batch, batch_recon_const, batch_recon_const_outres, batch_imgnames
  
  def save_output_with_gt(self, net_op, gt, epoch, itr_id, prefix, batch_size, num_cols=8, net_recon_const=None):
    net_out_img = self.save_output(net_op, batch_size, num_cols=num_cols, net_recon_const=net_recon_const)
    gt_out_img = self.save_output(gt, batch_size, num_cols=num_cols, net_recon_const=net_recon_const)
    num_rows = np.int_(np.ceil((batch_size*1.)/num_cols))
    border_img = 255*np.ones((num_rows*self.outshape[0], 128, 3), dtype='uint8')
    out_fn_pred = '%s/%s_pred_%06d_%06d.png' % (self.out_directory, prefix, epoch, itr_id)
    print('[DEBUG] Writing output image: %s' % out_fn_pred)
    cv2.imwrite(out_fn_pred, np.concatenate((net_out_img, border_img, gt_out_img), axis=1))
    
  def save_output(self, net_op, batch_size, num_cols=8, net_recon_const=None):
    num_rows = np.int_(np.ceil((batch_size*1.)/num_cols))
    out_img = np.zeros((num_rows*self.outshape[0], num_cols*self.outshape[1], 3), dtype='uint8')
    img_lab = np.zeros((self.outshape[0], self.outshape[1], 3), dtype='uint8')
    c = 0
    r = 0
    for i in range(batch_size):
      if(i % num_cols == 0 and i > 0):
        r = r + 1
        c = 0
      img_lab[..., 0] = self.__get_decoded_img(net_recon_const[i, ...].reshape(self.outshape[0], self.outshape[1]))
      img_lab[..., 1] = self.__get_decoded_img(net_op[i, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1]))
      img_lab[..., 2] = self.__get_decoded_img(net_op[i, np.prod(self.shape):].reshape(self.shape[0], self.shape[1]))
      img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
      out_img[r*self.outshape[0]:(r+1)*self.outshape[0], c*self.outshape[1]:(c+1)*self.outshape[1], ...] = img_rgb
      c = c+1
    return out_img
  
  def save_divcolor(self, net_op, gt, epoch, itr_id, prefix, batch_size, imgname, num_cols=8, net_recon_const=None):
    img_lab = np.zeros((self.outshape[0], self.outshape[1], 3), dtype='uint8')
    img_lab_mat = np.zeros((self.shape[0], self.shape[1], 2), dtype='uint8')
    if not os.path.exists('%s/%s' % (self.out_directory, imgname)):
      os.makedirs('%s/%s' % (self.out_directory, imgname))
    for i in range(batch_size):
      img_lab[..., 0] = self.__get_decoded_img(net_recon_const[i, ...].reshape(self.outshape[0], self.outshape[1]))
      img_lab[..., 1] = self.__get_decoded_img(net_op[i, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1]))
      img_lab[..., 2] = self.__get_decoded_img(net_op[i, np.prod(self.shape):].reshape(self.shape[0], self.shape[1]))
      img_lab_mat[..., 0] = 128.*net_op[i, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1])+128.
      img_lab_mat[..., 1] = 128.*net_op[i, np.prod(self.shape):].reshape(self.shape[0], self.shape[1])+128.
      img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
      out_fn_pred = '%s/%s/%s_%03d.png' % (self.out_directory, imgname, prefix, i)
      cv2.imwrite(out_fn_pred, img_rgb)
      out_fn_mat = '%s/%s/%s_%03d.mat' % (self.out_directory, imgname, prefix, i)
      np.save(out_fn_mat, img_lab_mat)
    img_lab[..., 0] = self.__get_decoded_img(net_recon_const[i, ...].reshape(self.outshape[0], self.outshape[1]))
    img_lab[..., 1] = self.__get_decoded_img(gt[0, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1]))
    img_lab[..., 2] = self.__get_decoded_img(gt[0, np.prod(self.shape):].reshape(self.shape[0], self.shape[1]))
    img_lab_mat[..., 0] = 128.*gt[0, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1])+128.
    img_lab_mat[..., 1] = 128.*gt[0, np.prod(self.shape):].reshape(self.shape[0], self.shape[1])+128.
    out_fn_pred = '%s/%s/gt.png' % (self.out_directory, imgname)
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(out_fn_pred, img_rgb)
    out_fn_mat = '%s/%s/gt.mat' % (self.out_directory, imgname)
    np.save(out_fn_mat, img_lab_mat)

  def __get_decoded_img(self, img_enc):
    img_dec = 128.*img_enc + 128
    img_dec[img_dec < 0.] = 0.
    img_dec[img_dec > 255.] = 255.
    return cv2.resize(np.uint8(img_dec), (self.outshape[0], self.outshape[1]))

  def __get_lossweights(self, img_vec):
    img_vec = img_vec*128.
    img_lossweights = np.zeros(img_vec.shape, dtype='f')
    img_vec_a = img_vec[:np.prod(self.shape)]
    binedges_a = self.binedges[0,...].reshape(-1)
    binid_a = [binedges_a.flat[np.abs(binedges_a-v).argmin()] for v in img_vec_a]
    img_vec_b = img_vec[np.prod(self.shape):]
    binedges_b = self.binedges[1,...].reshape(-1)
    binid_b = [binedges_b.flat[np.abs(binedges_b-v).argmin()] for v in img_vec_b]
    binweights = np.array([self.lossweights[v1][v2] for v1,v2 in zip(binid_a, binid_b)])
    img_lossweights[:np.prod(self.shape)] = binweights 
    img_lossweights[np.prod(self.shape):] = binweights
    return img_lossweights
