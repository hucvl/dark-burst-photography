# test.py
# description: Evaluation script for the set based model.

from __future__ import division
import os, scipy.io, time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import burst_nets
from skimage.measure import compare_ssim, compare_psnr
import dbputils
import cv2
import lpips_tf



n_burst = 1
input_dir = '../dataset/Fuji/short/'
gt_dir = '../dataset/Fuji/long/'
method = "burst_fuji"
d_id = 1 ## 1 for test, 2 for validation
result_name = "burst_fuji"
d_set = dbputils.d_set_for_id(d_id)
checkpoint_dir = '../checkpoint/Fuji/%s/'%method
result_dir = '../results/Fuji/%s/%s_%d/'%(d_set,result_name, n_burst)
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)

is_burst = True

# get test IDs
test_fns = glob.glob(gt_dir + '/%d*.RAF'%d_id)
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

misaligned = [10034, 10045, 10172]

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, None, 9])
in_image_low = tf.placeholder(tf.float32, [None, None, None, None, 9])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

coarse_outs = burst_nets.coarse_net(in_image_low,out_channels=9)
#out_image = burst_nets.fine_net(in_image, coarse_outs,out_channels=27)
out_image = burst_nets.fine_res_net(in_image, coarse_outs, out_channels=27)
distance_t = lpips_tf.lpips(gt_image, out_image, model='net-lin', net='alex')

t_vars = tf.trainable_variables()
saver = tf.train.Saver(t_vars)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)


if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


ssim_list = []
psnr_list = []
lpips_list = []
time_list = []
ratio_list = []
count = 0
	
for test_id in test_ids:
	in_files = glob.glob(input_dir + '%05d_00*.RAF' % test_id)
	for k in range(len(in_files)):
		## Read raw images.
		in_path = in_files[k]
		in_paths, complete = dbputils.get_burst_paths(in_path, n_burst=n_burst)
		in_fn = os.path.basename(in_path)
		gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % test_id)
		gt_path = gt_files[0]
		gt_fn = os.path.basename(gt_path)
		in_exposure = float(in_fn[9:-5])
		gt_exposure = float(gt_fn[9:-5])
		ratio = min(gt_exposure / in_exposure, 300)
		print(in_fn)

		## Pack and multiply with exp. ratio.
		inputs = []
		for in_path in in_paths:
			raw = rawpy.imread(in_path)
			input_full = dbputils.pack_fuji_raw(raw) * ratio
			input_full = np.expand_dims(input_full,0)
			input_full = np.minimum(input_full, 1.0)
			padded = np.zeros((input_full.shape[0],input_full.shape[1], input_full.shape[2]+38, input_full.shape[3]), input_full.dtype)
			padded[:,:,:input_full.shape[2],:] = input_full
			padded[:,:,padded.shape[2]-38:,:] = input_full[:,:,-38:,:]
			inputs.append(padded)
		inputs = np.array(inputs)
		inputs_low = dbputils.resize_samples(inputs, r=2)

		## Read gt.
		gt_raw = rawpy.imread(gt_path)
		gt_full = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
		gt_full = np.expand_dims(np.float32(gt_full / 65535.0), axis=0)
		gt_full_raw = np.expand_dims(dbputils.pack_fuji_raw(gt_raw), 0)
		
		print(inputs.shape, gt_full.shape)

		## Run model.
		st = time.time()
		if is_burst == True:
			st = time.time()
			output = sess.run(out_image, feed_dict={in_image: inputs, in_image_low: inputs_low,  gt_image: gt_full})
		else:
			outs = []
			for i in range(len(inputs)):
				output = sess.run(out_image, feed_dict={in_image: np.expand_dims(inputs[i], 0), in_image_low: np.expand_dims(inputs_low[i], 0)})
				output = np.minimum(np.maximum(output, 0), 1)
				outs.append(output)
			outs = np.array(outs)
			output = np.mean(outs, 0)
		time_ = time.time() - st
		
		## Compute lpips
		output = output[:,:,:gt_full.shape[-2],:]
		lpips_val = sess.run([distance_t], feed_dict={out_image: output, gt_image: gt_full})
		lpips_val = lpips_val[0][0]
		output = np.minimum(np.maximum(output, 0), 1)
		output = output[0, :,  :, :]
		gt_full = gt_full[0, :, :, :]
		print(output.shape, gt_full.shape)
			
		## Compute psnr, ssim
		if test_id not in misaligned:
			ssim_val = compare_ssim(output, gt_full, multichannel=True)
			psnr_val = compare_psnr(output, gt_full)
			count += 1
			ssim_list.append(ssim_val)
			psnr_list.append(psnr_val)
			lpips_list.append(lpips_val)
			time_list.append(time_)
			ratio_list.append(ratio)
			
			avg_ssim = np.mean(ssim_list)
			avg_psnr = np.mean(psnr_list)
			avg_lpips = np.mean(lpips_list)
			avg_time = np.mean(time_list[1:])

			print("Avg ssim: %.4f, Avg psnr: %.4f, Avg lpips: %.4f, Time elapsed: %.3f, %d" % (avg_ssim, avg_psnr, avg_lpips, time_, count))
			
		## Save results to an array.
		quant_results = np.array(ssim_list, dtype=np.float32)
		quant_results = np.expand_dims(quant_results, 0)
		quant_results = np.concatenate([quant_results, np.expand_dims(np.array(psnr_list, dtype=np.float32), 0)], 0)
		quant_results = np.concatenate([quant_results, np.expand_dims(np.array(lpips_list, dtype=np.float32), 0)], 0)
		quant_results = np.concatenate([quant_results, np.expand_dims(np.array(ratio_list, dtype=np.float32), 0)], 0)
		quant_results = np.concatenate([quant_results, np.expand_dims(np.array(time_list, dtype=np.float32), 0)], 0)
		np.save(result_dir+"results.npy", quant_results)

		## Save output images.
		temp = cv2.cvtColor(np.uint8(output*255), cv2.COLOR_RGB2BGR)
		cv2.imwrite(result_dir+"%05d_00_%d_out.jpg"%(test_id, ratio), temp)
