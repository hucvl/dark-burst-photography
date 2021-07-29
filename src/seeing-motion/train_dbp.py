from __future__ import division
import os,time
import numpy as np
import pdb
import glob
import tensorflow as tf
import math
from skimage.measure import compare_ssim, compare_psnr
import cv2
import sys

#from network import *
sys.path.insert(0, "..")
from prefetch_queue_shuffle import *
from vgg import *
import burst_nets
from network import *

from CX.CX_helper import *
from CX.config import *

data_dir = "../../dataset/DRV/"

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]#[:5]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]

save_freq = 100


method = "burst_l1_drv_full"
checkpoint_dir = './checkpoints/%s'%method
ps = 918
n_burst = 8
in_image = tf.placeholder(tf.float32, [None, 1, ps, ps, 4])
in_image_low = tf.placeholder(tf.float32, [None, 1, ps//2, ps//2, 4])
gt_image = tf.placeholder(tf.float32, [1, ps, ps, 3])
coarse_outs = burst_nets.coarse_net(in_image_low)
out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=3, demosaic=False)


vgg_fake = build_vgg19(255*out_image)
vgg_real = build_vgg19(255*gt_image, reuse=True)


# #CX Loss.
CX_loss_list = [CX_loss_helper(vgg_real[layer], vgg_fake[layer], config.CX)
                        for layer, w in config.CX.feat_layers.items()]
CX_loss = tf.reduce_sum(CX_loss_list)
CX_loss = CX_loss/255.

G_l1 = tf.reduce_mean(tf.abs(gt_image - out_image))
G_loss = G_l1 + CX_loss

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

sess=tf.Session()
t_vars = tf.trainable_variables()

restore_variables = []
for t in t_vars:
    print(t.name)
    if "g_conv10_fine" not in t.name:
        print("Restoring %s" % t.name)
        restore_variables.append(t)
saver = tf.train.Saver(restore_variables)


sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

loss_sum = tf.summary.scalar('loss',G_loss)
sum_writer = tf.summary.FileWriter('./log',sess.graph)


saver = tf.train.Saver(t_vars)

counter = 1

num_workers = 1
load_fn = load_fn_burst
p_queue = PrefetchQueue(load_fn, train_ids, 1, 32, num_workers=num_workers)

learning_rate = 1e-4
num_epochs = 251
for epoch in range(0,num_epochs):
    if epoch > num_epochs//2:
        learning_rate = 1e-5

    for ind in range(len(train_ids)):
        st = time.time()
        X = p_queue.get_batch()  #load a batch for training

        inputs = X[0]
        inputs_low = X[1]
        gt_np = X[2]

        print(inputs.shape, inputs_low.shape, gt_np.shape)
        r = np.random.randint(1,n_burst+1)
        inputs, inputs_low = inputs[:r, :,:,:], inputs_low[:r, :,:,:]
        print(inputs.shape, inputs_low.shape, gt_np.shape)

        _,G_current,out_np,sum_str=sess.run([G_opt,G_loss,out_image,loss_sum],feed_dict={in_image: inputs, in_image_low: inputs_low, gt_image:gt_np, lr:learning_rate})

        out_np = np.minimum(np.maximum(out_np,0),1)


        sum_writer.add_summary(sum_str,counter)
        counter += 1


        print("%d %s Loss=%.3f Time=%.3f"%(epoch,ind,G_current, time.time()-st))

        if epoch%save_freq==0:
          #save results for visualization
            if not os.path.isdir("result/%04d"%epoch):
                os.makedirs("result/%04d"%epoch)
            temp = np.concatenate((out_np[0,:,:,:],gt_np[0,:,:,:]),axis=1)*255
            temp = np.clip(temp,0,255)
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            cv2.imwrite("result/%04d/train_%s.jpg"%(epoch,ind),np.uint8(temp))

    if (epoch % save_freq == 0) and epoch > 0:
        saver.save(sess,"%s/model.ckpt" % checkpoint_dir)
