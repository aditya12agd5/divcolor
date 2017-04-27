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
from data_loaders.lab_imageloader import lab_imageloader
from arch.vae_skipconn import vae_skipconn as vae
#from arch.vae_wo_skipconn import vae_wo_skipconn as vae

from arch.network import network as network

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
flags.DEFINE_boolean("is_train", True, "Is training flag") 
flags.DEFINE_integer("hidden_size", 64, "size of the hidden VAE unit")
flags.DEFINE_float("lr_vae", 1e-6, "learning rate for vae")
flags.DEFINE_integer("max_epoch_vae", 10, "max epoch")
flags.DEFINE_integer("pc_comp", 20, "number of principle components")


FLAGS = flags.FLAGS

def main():
  if(len(sys.argv) == 1):
    raise NameError('[ERROR] No dataset key')
  elif(sys.argv[1] == 'lfw'):
    FLAGS.updates_per_epoch = 380
    FLAGS.log_interval = 120
    FLAGS.out_dir = 'data/output/lfw/'
    FLAGS.list_dir = 'data/imglist/lfw/'
    FLAGS.pc_dir = 'data/pcomp/lfw/'
  #add other datasets here
  else:
    raise NameError('[ERROR] Incorrect dataset key')

  data_loader = lab_imageloader(FLAGS.in_dir, \
    os.path.join(FLAGS.out_dir, 'images'), \
    listdir=FLAGS.list_dir)

 #Train colorfield VAE
  graph_vae = tf.Graph()
  with graph_vae.as_default():
    model_colorfield = vae(FLAGS, nch=2)
    dnn = network(model_colorfield, data_loader, 2, FLAGS)
    latent_vars_colorfield, latent_vars_colorfield_musigma_test = \
     dnn.train_vae(os.path.join(FLAGS.out_dir, 'models'), FLAGS.is_train)
 
  np.save(os.path.join(FLAGS.out_dir, 'lv_color_train.mat'), latent_vars_colorfield)
  np.save(os.path.join(FLAGS.out_dir, 'lv_color_test.mat'), latent_vars_colorfield_musigma_test)

if __name__ == "__main__":
  main()
