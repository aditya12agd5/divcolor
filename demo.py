import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import socket
import sys

import tensorflow as tf
import numpy as np
from vae.data_loaders.lab_imageloader import lab_imageloader
from vae.arch.vae_skipconn import vae_skipconn as vae
from vae.arch.network import network
from third_party.save_zhang_feats import save_zhang_feats

flags = tf.flags

#Directory params
flags.DEFINE_string("out_dir", "", "")
flags.DEFINE_string("in_dir", "", "")
flags.DEFINE_string("list_dir", "", "")

#Dataset Params
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1, "number of updates per epoch")
flags.DEFINE_integer("log_interval", 1, "input image height")
flags.DEFINE_integer("img_width", 64, "input image width")
flags.DEFINE_integer("img_height", 64, "input image height")

#Network Params
flags.DEFINE_boolean("is_only_data", False, "Is training flag") 
flags.DEFINE_boolean("is_train", False, "Is training flag") 
flags.DEFINE_boolean("is_run_cvae", False, "Is training flag") 
flags.DEFINE_integer("hidden_size", 64, "size of the hidden VAE unit")
flags.DEFINE_float("lr_vae", 1e-6, "learning rate for vae")
flags.DEFINE_integer("max_epoch_vae", 10, "max epoch")
flags.DEFINE_integer("pc_comp", 20, "number of principle components")

FLAGS = flags.FLAGS

def main():
  
  FLAGS.log_interval = 1
  FLAGS.list_dir = None 
  FLAGS.in_dir = 'data/testimgs/'
  data_loader = lab_imageloader(FLAGS.in_dir, \
    'data/output/testimgs', listdir=None)
  img_fns = data_loader.test_img_fns
  
  if(FLAGS.is_only_data == True):
    feats_fns = save_zhang_feats(img_fns)
  
    with open('%s/list.train.txt' % FLAGS.in_dir, 'w') as fp:
     for feats_fn in feats_fns:
       fp.write('%s\n' % feats_fn)

    with open('%s/list.test.txt' % FLAGS.in_dir, 'w') as fp:
     for feats_fn in feats_fns:
       fp.write('%s\n' % feats_fn)
    
    np.save('%s/lv_color_train.mat.npy' % FLAGS.in_dir, \
      np.zeros((len(img_fns), 2*FLAGS.hidden_size)))
    np.save('%s/lv_color_test.mat.npy' % FLAGS.in_dir, \
     np.zeros((len(img_fns), 2*FLAGS.hidden_size)))
  else:
    nmix = 8
    lv_mdn_test = np.load(os.path.join(FLAGS.in_dir, 'lv_color_mdn_test.mat.npy'))
    num_batches = np.int_(np.ceil((lv_mdn_test.shape[0]*1.)/FLAGS.batch_size))
    latent_vars_colorfield_test = np.zeros((0, FLAGS.hidden_size), dtype='f')
    for i in range(lv_mdn_test.shape[0]):
      curr_means = lv_mdn_test[i, :FLAGS.hidden_size*nmix].reshape(nmix, FLAGS.hidden_size)
      curr_sigma = lv_mdn_test[i, FLAGS.hidden_size*nmix:(FLAGS.hidden_size+1)*nmix].reshape(-1)
      curr_pi = lv_mdn_test[i, (FLAGS.hidden_size+1)*nmix:].reshape(-1)
      selectid = np.argsort(-1*curr_pi)
      curr_sample = np.tile(curr_means[selectid, ...], (np.int_(np.round((FLAGS.batch_size*1.)/nmix)), 1))
      latent_vars_colorfield_test = \
        np.concatenate((latent_vars_colorfield_test, curr_sample), axis=0)
 
    graph_divcolor = tf.Graph()
    with graph_divcolor.as_default():
      model_colorfield = vae(FLAGS, nch=2, condinference_flag=True)
      dnn = network(model_colorfield, data_loader, 2, FLAGS)
      dnn.run_divcolor('data/imagenet_models/' , \
        latent_vars_colorfield_test, num_batches=num_batches)
  
if __name__ == "__main__":
  main()
