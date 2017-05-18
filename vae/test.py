import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import socket
import sys

import tensorflow as tf
import numpy as np
from data_loaders.lab_imageloader import lab_imageloader
from arch.vae_skipconn import vae_skipconn as vae
#from arch.vae_wo_skipconn import vae_wo_skipconn as vae

from arch.network import network

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
flags.DEFINE_boolean("is_run_cvae", False, "Is training flag") 
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
  else:
    raise NameError('[ERROR] Incorrect dataset key')
  data_loader = lab_imageloader(FLAGS.in_dir, \
    os.path.join(FLAGS.out_dir, 'images'), \
    listdir=FLAGS.list_dir)

  #Diverse Colorization
  nmix = 8
  num_batches = 31
  lv_mdn_test = np.load(os.path.join(FLAGS.out_dir, 'lv_color_mdn_test.mat.npy'))
   
  graph_divcolor = tf.Graph()
  with graph_divcolor.as_default():
    model_colorfield = vae(FLAGS, nch=2, condinference_flag=True)
    dnn = network(model_colorfield, data_loader, 2, FLAGS)
    dnn.run_divcolor(os.path.join(FLAGS.out_dir, 'models') , \
      latent_vars_colorfield_test, num_batches=num_batches)
    if(FLAGS.is_run_cvae == True):
      dnn.run_cvae(os.path.join(FLAGS.out_dir, 'models') , \
       lv_mdn_test, num_batches=num_batches, num_repeat=8, num_cluster=5)
  
if __name__ == "__main__":
  main()
