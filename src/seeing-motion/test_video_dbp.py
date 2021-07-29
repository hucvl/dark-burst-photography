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

sys.path.insert(0, "..")
from prefetch_queue_shuffle import *
import burst_nets
from network import *

import flow_tools

data_dir = "../../dataset/DRV/"

n_burst = 3
#method = "drv"
method = "burst_l1_drv_full"
checkpoint_dir = './checkpoints/%s' % method
use_flow = True

is_original = False

if is_original == True:
	use_flow = False
	n_burst = 1

if use_flow == True:
	result_dir = './results/%s_flow_%d/'%(method, n_burst)
else:
	result_dir = './results/%s_%d/'%(method, n_burst)


if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(result_dir+"video"):
    os.makedirs(result_dir+"video")
if not os.path.isdir(result_dir+"frames"):
    os.makedirs(result_dir+"frames")

print(result_dir)

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]

# sess=tf.Session()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if is_original == True:
	in_image=tf.placeholder(tf.float32,[None,None,None,3])
	gt_image=tf.placeholder(tf.float32,[None,None,None,3])

	out_image=Unet(in_image)

else:
	in_image = tf.placeholder(tf.float32, [None, 1, 918, 1374, 4])
	in_image_low = tf.placeholder(tf.float32, [None, 1, 459, 687, 4])
	in_flows = tf.placeholder(tf.float32, [None, 1, 459, 687, 2])
	gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

	coarse_outs = burst_nets.coarse_net(in_image_low)

	if use_flow == False:
		out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=3, demosaic=False)
	else:
		#out_image = burst_nets.fine_res_net_flow(in_image, coarse_outs, in_flows, out_channels=3, demosaic=False)
		out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=3, demosaic=False)

sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded '+ckpt.model_checkpoint_path)
	saver.restore(sess,ckpt.model_checkpoint_path)

# test dynamic videos
print(result_dir)
test_ids = np.arange(23)
test_ids = [1,7,10,11,17]
for test_id in test_ids:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	video = cv2.VideoWriter(result_dir+'video/M%04d.avi'%test_id,fourcc, 20.0, (1374,918))

	in_files = sorted(glob.glob(data_dir + 'VBM4D_rawRGB/M%04d/*.png'%test_id))
	if not os.path.isdir(result_dir+"frames/M%04d"%test_id):
		os.makedirs(result_dir+"frames/M%04d"%test_id)
	for k in range(2,len(in_files)-n_burst):
	#for k in range(4,10):

		print('running %s-th sequence %d-th frame...'%(test_id,k))
		inputs = []
		inputs_low = []
		for b in range(n_burst):
			in_path = in_files[k+b]
			_, in_fn = os.path.split(in_path)

			im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
			im = np.float32(im/65535.0)

			if is_original == False:
				im_replicated = np.zeros(shape=(im.shape[0], im.shape[1], im.shape[2]+1))
				im_replicated[:,:,0] = im[:,:,2]
				im_replicated[:,:,1] = im[:,:,1]
				im_replicated[:,:,2] = im[:,:,1]
				im_replicated[:,:,3] = im[:,:,0]
			
			else:
				im_replicated = im

			im_low = cv2.resize(im_replicated, (im_replicated.shape[1]//2, im.shape[0]//2))
			in_np = np.expand_dims(im_replicated,axis = 0)
			in_np_low = np.expand_dims(im_low, 0)

			inputs.append(in_np)
			inputs_low.append(in_np_low)

		inputs = np.array(inputs)
		inputs_low = np.array(inputs_low)
		print(inputs.shape, inputs_low.shape)

		if is_original == True:
			out_np = sess.run(out_image,feed_dict={in_image: inputs[0]})
			
		else:
			if use_flow == False:
				out_np = sess.run(out_image,feed_dict={in_image: inputs, in_image_low: inputs_low})
			else:
				coarse_outs_np = sess.run(coarse_outs,feed_dict={in_image_low: inputs_low})
				coarse_outs_np = np.clip(coarse_outs_np, 0, 1)
				inputs, coarse_outs_np = flow_tools.warp_frames_with_masking(inputs, coarse_outs_np, result_dir+"frames/M%04d/"%test_id, k+1)
				out_np = sess.run(out_image,feed_dict={in_image: inputs, coarse_outs: coarse_outs_np})

		if is_original == False:
			out_np = out_np[:,:,:,::-1]

		out_np = np.minimum(np.maximum(out_np,0),1)
		out_np = out_np[0,:,:,:]
		out = np.uint8(out_np*255.0)
		if is_original == True:
			cv2.imwrite(result_dir+"frames/M%04d/%04d.jpg"%(test_id,k),out)
		cv2.imwrite(result_dir+"frames/M%04d/%04d.jpg"%(test_id,k+1),out)
		video.write(out)
	video.release()
