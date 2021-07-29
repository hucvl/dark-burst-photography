# train.py
# description: Training script for Burst Photography for Learning to Enhance Extremely Dark Images.

## Train coarse network with ps=256, 4000 epochs
## Train fine network with ps=512, ps_low=256, 4000 epochs without fixed params, lr=1e-4 -> 1e-5 after 2000 epochs.
## Train set-based fine network with ps_fine=512, ps_denoise=256, 1000 epochs with fixed coarse params.

from __future__ import division
import os, time
import tensorflow as tf
import numpy as np
import rawpy
import glob
import dbputils
import burst_nets
import cv2
from vgg import *
from CX.CX_helper import *
from CX.config import *

import time

input_dir = '/../../dataset/Fuji/short/'
gt_dir = '/../../dataset/Fuji/long/'
result_dir = "../results/Fuji/train/"
method_name = "burst_fuji"
checkpoint_dir = '../checkpoint/Fuji/%s/' % method_name
result_dir = "../results/learning/%s" % method_name

# get train IDs
train_fns = glob.glob(gt_dir + '0*.RAF')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

save_freq = 50
train_coarse = False
fix_coarse_weights = True ## fixes the weights of coarse network.
finetune = True ## all weights are already initialized from ckpt if True.

sess = tf.Session()
n_burst = 8
max_burst = n_burst
ps = 512
ps_low = int(ps/2)
raw_ratio = 3


## Losses.
########################
if train_coarse == True:
	in_image = tf.placeholder(tf.float32, [None, 1, ps_low, ps_low, 9])
	in_image_low = tf.placeholder(tf.float32, [None, 1, ps_low//2, ps_low//2, 9])
	gt_image_low = tf.placeholder(tf.float32, [1, ps_low, ps_low, 9])
	coarse_outs = burst_nets.coarse_net(in_image, out_channels=9)
	out_image = coarse_outs[0]
	G_loss_raw = tf.map_fn(lambda x: tf.reduce_mean(tf.abs(gt_image_low - x)), coarse_outs)
	G_loss_raw = tf.reduce_mean(G_loss_raw)
	G_loss =  G_loss_raw

else:
	in_image = tf.placeholder(tf.float32, [None, 1, ps, ps, 9])
	in_image_low = tf.placeholder(tf.float32, [None, 1, ps_low, ps_low, 9])
	gt_image = tf.placeholder(tf.float32, [1, ps*3, ps*3, 3])
	gt_image_low = tf.placeholder(tf.float32, [None, ps_low, ps_low, 9])
	coarse_outs = burst_nets.coarse_net(in_image_low, out_channels=9)
	out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=27)

	G_l1 = tf.reduce_mean(tf.abs(gt_image - out_image))
	G_pix = G_l1

	# vgg_fake = build_vgg19(255*out_image[:,:,:,::-1])
	# vgg_real = build_vgg19(255*gt_image[:,:,:,::-1], reuse=True)

	## CX Loss.
	CX_loss_list = [CX_loss_helper(vgg_real[layer], vgg_fake[layer], config.CX)
							for layer, w in config.CX.feat_layers.items()]
	CX_loss = tf.reduce_sum(CX_loss_list)
	G_feat = CX_loss/255.
	
	## Perceptual Loss
	# per_loss_list = [tf.reduce_mean(tf.abs(vgg_real[layer]-vgg_fake[layer])) for layer, w in config.CX.feat_layers.items()]
	# per_loss = tf.reduce_mean(per_loss_list)/255.
	# G_feat += 0.1*per_loss

	G_loss_hd = 0.1*G_feat + G_pix
	if fix_coarse_weights == False:
		G_loss_raw = tf.map_fn(lambda x: tf.reduce_mean(tf.abs(x-gt_image_low)), coarse_outs)
		G_loss_raw = tf.reduce_mean(G_loss_raw)	
		G_loss = G_loss_raw + G_loss_hd
	else:
		G_loss = G_loss_hd


## Variable Optimization.
########################
t_vars = tf.trainable_variables()
restore_variables = []
new_variables = []
new_kws = []
if train_coarse == False:
	new_kws.append("fine")
for t in t_vars:
	is_old = True
	for kw in new_kws:
		if kw in t.name:
			is_old = False
			break
	if is_old == True:
		print(t.name)
		restore_variables.append(t)
	else:
		print("new: %s" % t.name)
		new_variables.append(t)

variables = []
lr = tf.placeholder(tf.float32)
variables = new_variables + restore_variables
if fix_coarse_weights == True:
	G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=new_variables)
else:
	G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=variables)

if(finetune == False):
	print("Restoring coarse vars..")
	saver = tf.train.Saver(restore_variables)
else:
	print("Restoring all vars..")
	saver = tf.train.Saver(t_vars)

sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

if(len(new_variables) > 0 and finetune == False):
	## Initialize new vars from scratch.
	sess.run(tf.initialize_variables(new_variables))
saver = tf.train.Saver(t_vars)

gt_images = [None] * 6000
gt_image_raws = [None] * 6000
input_images = []
for i in range(max_burst):
	d = {}
	d['300'] = [None] * len(train_ids)
	d['250'] = [None] * len(train_ids)
	d['100'] = [None] * len(train_ids)
	input_images.append(d)
# g_loss = np.zeros((5000, 1))
g_loss_raw = np.zeros((5000, 1))
g_loss_hd = np.zeros((5000, 1))
g_loss_l1 = np.zeros((5000, 1))
g_loss_feat = np.zeros((5000, 1))

## Read data.
########################
def load_input_output(idx, nb, ps=512):
	## file path for training id.
	st_init = time.time()
	train_id = train_ids[idx]
	in_files = glob.glob(input_dir + '%05d_00*.RAF' % train_id)
	in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
	in_fn = os.path.basename(in_path)
	
	## file paths for the burst and exposure ratio.
	in_paths, complete = dbputils.get_burst_paths(in_path, max_burst)
	gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % train_id)
	gt_path = gt_files[0]
	gt_fn = os.path.basename(gt_path)
	in_exposure = float(in_fn[9:-5])
	gt_exposure = float(gt_fn[9:-5])
	ratio = min(gt_exposure / in_exposure, 300)

	## read burst images
	if input_images[0][str(ratio)[0:3]][idx] is None:
		for k in range(len(in_paths)):
			in_path = in_paths[k]
			if os.path.isfile(in_path):
				raw = rawpy.imread(in_path)
				raw = dbputils.pack_fuji_raw(raw)*ratio
				if train_coarse == True:
					raw = dbputils.resize(raw, r=2)
				raw = np.expand_dims(raw, 0)
				raw = np.minimum(raw, 1.0)
				input_images[k][str(ratio)[0:3]][idx] = raw

		## read gt.
		gt_raw = rawpy.imread(gt_path)
		im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
		im = np.float32(im/65535.0)
		raw_im = dbputils.pack_fuji_raw(gt_raw)
		if train_coarse == True:
			im = dbputils.resize(im, r=2)
			raw_im = dbputils.resize(raw_im, r=2)

		im = np.expand_dims(im, axis=0)
		raw_im = np.expand_dims(raw_im, 0)
		im = np.minimum(im, 1.0)
		raw_im = np.minimum(raw_im, 1.0)
		gt_images[idx] = im
		gt_image_raws[idx] = raw_im

	## get inputs and output
	gt_full = gt_images[idx]
	gt_full_raw = gt_image_raws[idx]
	input_patches = []
	for k in range(len(in_paths)):
		input_full = input_images[k][str(ratio)[0:3]][idx]
		input_patches.append(input_full)

	## preprocessing
	st = time.time()
	input_patches, gt_patch, gt_patch_raw = dbputils.crop_samples(input_patches, gt_full, gt_full_raw, ps=ps, raw_ratio=raw_ratio)
	input_patches, gt_patch, gt_patch_raw = dbputils.augment_samples(input_patches, gt_patch, gt_patch_raw)
	st_resize = time.time()
	input_patches_low = dbputils.resize_samples(input_patches)


	if train_coarse == False:
		gt_patch_raw = np.expand_dims(dbputils.resize(gt_patch_raw[0, :, :, :], r=2), 0)
	
	return input_patches[:nb, :,:,:], input_patches_low[:nb, :,:,:], gt_patch, gt_patch_raw, ratio

learning_rate = 1e-4
print("Starting training..")
start_epoch = 1700
num_epochs = 4001

## Training.
########################
for epoch in range(start_epoch, num_epochs):
	cnt = 0
	if epoch > int(num_epochs)*1/2:
		learning_rate = 1e-5

	st = time.time()

	if epoch % save_freq == 0 and epoch > 0:
		print("Saving model..")
		saver.save(sess, checkpoint_dir + 'model.ckpt')

	for idx in np.random.permutation(len(train_ids)):
		r = np.random.randint(1,n_burst+1)
		if train_coarse == True:
			input_patches, input_patches_low, gt_patch, gt_patch_raw, ratio = load_input_output(idx,r,ps=ps_low)
			_, G_current_raw, output = sess.run([G_opt, G_loss_raw, out_image], feed_dict={in_image: input_patches, gt_image_low: gt_patch_raw, lr: learning_rate})
			output = np.minimum(np.maximum(output, 0), 1)
			g_loss_raw[idx] = G_current_raw
			print("%d %d  RAW Loss=%.4f Time=%.4f" % (epoch, cnt, np.mean(g_loss_raw[np.where(g_loss_raw)]), time.time() - st))
			cnt += 1

		else:
			input_patches, input_patches_low, gt_patch, gt_patch_raw, ratio = load_input_output(idx,r,ps=ps)

			_, G_current_hd, output = sess.run([G_opt, G_loss_hd, out_image], feed_dict={in_image: input_patches, in_image_low: input_patches_low, gt_image: gt_patch, gt_image_low: gt_patch_raw, lr: learning_rate})

			output = np.minimum(np.maximum(output, 0), 1)
			g_loss_hd[idx] = G_current_hd
			print("%d %d Loss=%.4f Time=%.4f" % (epoch, cnt, np.mean(g_loss_hd[np.where(g_loss_hd)]), time.time() - st))
			
			cnt += 1

		if epoch % save_freq == 0:
			if not os.path.isdir("%s/%04d"%(result_dir, epoch)):
				os.makedirs("%s/%04d"%(result_dir, epoch))

			if train_coarse == False:
				temp = np.concatenate((output[0,:,:,:],gt_patch[0,:,:,:]),axis=1)*255
				temp = np.clip(temp,0,255)
				cv2.imwrite("%s/%04d/train_%s.jpg"%(result_dir, epoch,idx),np.uint8(temp))