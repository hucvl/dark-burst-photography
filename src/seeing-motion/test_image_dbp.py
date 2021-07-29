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
from vgg import *
from network import *
import burst_nets
import lpips_tf


sys.path.insert(0, "../RAFT/")
sys.path.insert(0, "../RAFT/core/")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import torch
from PIL import Image
import argparse

data_dir = "../../dataset/DRV/"

n_burst = 3
#method = "drv"
method = "burst_l1_drv_full"
checkpoint_dir = './checkpoints/%s' % method
result_dir = './results/%s_%d/'%(method, n_burst)

is_original = False

if is_original == False:
    in_image = tf.placeholder(tf.float32, [None, None, None, None, 4])
    in_image_low = tf.placeholder(tf.float32, [None, None, None, None, 4])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
    coarse_outs = burst_nets.coarse_net(in_image_low)
    # coarse_outs_warped = tf.placeholder(tf.float32, [None, None, None, None, 4])
    out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=3, demosaic=False)

else:
    in_image=tf.placeholder(tf.float32,[None,None,None,3])
    gt_image=tf.placeholder(tf.float32,[None,None,None,3])
    out_image=Unet(in_image)

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]

test_ids = ["0001"]

distance_t = lpips_tf.lpips(gt_image, out_image, model='net-lin', net='alex')

sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)



psnr_list = []
ssim_list = []
lpips_list = []
for test_id in test_ids:
    in_files = sorted(glob.glob(data_dir + 'VBM4D_rawRGB/%s/*.png'%test_id))
    gt_files = glob.glob(data_dir + 'long/%s/half0001*.png'%test_id)
    gt_path = gt_files[0]
    im = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    gt_np = np.float32(im/65535.0)

    inputs = []
    inputs_low = []
    for k in range(4,n_burst+4):

        print('running %s-th sequence %d-th frame...'%(test_id,k))
        in_path = in_files[k]
        im = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        _, in_fn = os.path.split(in_path)

        im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        im = np.float32(im/65535.0)

        im_replicated = np.zeros(shape=(im.shape[0], im.shape[1], im.shape[2]+1))
        im_replicated[:,:,0] = im[:,:,2]
        im_replicated[:,:,1] = im[:,:,1]
        im_replicated[:,:,2] = im[:,:,1]
        im_replicated[:,:,3] = im[:,:,0]
        
        if is_original == True:
            im_replicated = im

        im_low = cv2.resize(im_replicated, (im_replicated.shape[1]//2, im.shape[0]//2))
        in_np = np.expand_dims(im_replicated,axis = 0)
        in_np_low = np.expand_dims(im_low, 0)

        inputs.append(in_np)
        inputs_low.append(in_np_low)

    inputs = np.array(inputs)
    inputs_low = np.array(inputs_low)

    if is_original == False:
        coarse_outs_np = sess.run(coarse_outs,feed_dict={in_image_low: inputs_low})
        coarse_outs_np = np.clip(coarse_outs_np, 0, 1)
        out_np = sess.run(out_image,feed_dict={in_image: inputs, in_image_low: inputs_low})

    else:
        out_np = sess.run(out_image,feed_dict={in_image: inputs[0]})
        out_np = out_np[:,:,:,::-1]

    lpips_val = sess.run([distance_t], feed_dict={out_image: out_np, gt_image: np.expand_dims(gt_np,0)})
    lpips_val = lpips_val[0][0]

    out_np = np.clip(out_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)

    out_np = out_np[0,:,:,:]
    out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    print(inputs.shape, out_np.shape, gt_np.shape)
    psnr_val = compare_psnr(out_np, gt_np)
    ssim_val = compare_ssim(out_np, gt_np, multichannel=True)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    lpips_list.append(lpips_val)
    print("Avg psnr: %.4f, Avg ssim: %.4f, Avg lpips: %.4f, %d" % (np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), len(psnr_list)))


    out_np = np.uint8(out_np*255)
    gt_np = np.uint8(gt_np*255)

    cv2.imwrite(result_dir+"%s_out.png"%(test_id), out_np)
