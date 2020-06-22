# freeze_graph.py
# description: Script for freezing the graph from checkpoint.

import burst_nets
import tensorflow as tf


method = "burst_l1_cx"
checkpoint_dir = '../checkpoint/Sony/%s/'%method

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, None, 4])
in_image_low = tf.placeholder(tf.float32, [None, None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

coarse_outs = burst_nets.coarse_net(in_image_low)
out_image = burst_nets.fine_net(in_image, coarse_outs)

t_vars = tf.trainable_variables()
saver = tf.train.Saver(t_vars)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
tf.train.write_graph(sess.graph.as_graph_def(), '.', checkpoint_dir + 'model.pbtxt', as_text=True)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(checkpoint_dir + "model.pbtxt", "", False, 
                          checkpoint_dir + "model.ckpt", "DepthToSpace",
                           "save/restore_all", "save/Const:0",
                          checkpoint_dir + "frozen_model.pb", True, "")