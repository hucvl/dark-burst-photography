# test.py
# description: Evaluation script for the set based model.

import tensorflow as tf
import rawpy, glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
import utils, time



n_burst = 1
input_dir = '../dataset/Sony/short/'
gt_dir = '../dataset/Sony/long/'
method = "burst_l1_cx"
d_id = 1 ## 1 for test, 2 for validation
result_name = "burst_l1_cx"
d_set = utils.d_set_for_id(d_id)
checkpoint_dir = '../checkpoint/Sony/%s/'%method
result_dir = '../results/%s/%s/'%(d_set,result_name)
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)

is_burst = False

# get test IDs
test_fns = glob.glob(gt_dir + '/%d*.ARW'%d_id)
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

checkpoint_dir = "../checkpoint/Sony/burst_l1_cx/"
graph_path = checkpoint_dir + "frozen_model.pb"
sess = tf.Session()

print("load graph")
with gfile.FastGFile(graph_path,'rb') as f:
    graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
sess.graph.as_default()
tf.import_graph_def(graph_def, name='')

[print(n.name) for n in tf.get_default_graph().as_graph_def().node]


ssim_list = []
psnr_list = []
lpips_list = []
time_list = []
ratio_list = []
count = 0
	
for test_id in test_ids:
	in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
	for k in range(len(in_files)):
		## Read raw images.
		in_path = in_files[k]
		in_paths, complete = utils.get_burst_paths(in_path, n_burst=n_burst)
		in_fn = os.path.basename(in_path)
		gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
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
			input_full = utils.pack_raw(raw) * ratio
			input_full = np.expand_dims(input_full,0)
			input_full = np.minimum(input_full, 1.0)
			inputs.append(input_full)
		inputs = np.array(inputs)
		inputs_low = utils.resize_samples(inputs)

		## Read gt.
		gt_raw = rawpy.imread(gt_path)
		gt_full = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
		gt_full = np.expand_dims(np.float32(gt_full / 65535.0), axis=0)
		gt_full_raw = np.expand_dims(utils.pack_raw(gt_raw), 0)
		
		## Run model.
		st = time.time()
		
		print(inputs.shape, inputs_low.shape)
		output = sess.run("DepthToSpace:0", feed_dict={"Placeholder:0": inputs, "Placeholder_1:0": inputs_low})
		print(output.shape)
		output = output[0]

		time_ = time.time() - st

		## Save output images.
		temp = cv2.cvtColor(np.uint8(output*255), cv2.COLOR_RGB2BGR)
		cv2.imwrite(result_dir+"%05d_00_%d_out.jpg"%(test_id, ratio), temp)
