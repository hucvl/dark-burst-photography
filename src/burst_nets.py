# set_nets.py: Ahmet Serdar Karadeniz
# description: Networks for set based model.

from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
import numpy as np
from tensorflow.contrib.layers import layer_norm, instance_norm
import math
import dbputils


def lrelu(x):
	return tf.maximum(x * 0.2, x)

def upsample(input, s=2, nn=False):
	sh = tf.shape(input)
	newShape = s * sh[1:3]
	if nn == False:
		output = tf.image.resize_bilinear(input, newShape) 
	else:
		output = tf.image.resize_nearest_neighbor(input, newShape)
	return output

def upsample_and_concat(x1, x2, output_channels, in_channels, is_fine=False, block_idx=0):
	pool_size = 2
	if is_fine == True: 
		name = "deconv_fine_0"
		if block_idx > 0:
			name = name+"_%d"%block_idx
		with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
			deconv_filter = tf.get_variable(name=name,initializer=tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
		
		res_shape = pool_size*tf.shape(x1)[1:3]
		output_shape = tf.stack([tf.shape(x1)[0], res_shape[0], res_shape[1], output_channels])
		deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
		deconv_output = tf.concat([deconv, x2], 3)
	else:
		name = "Variable"
		if block_idx > 0:
			name = name+"_%d"%block_idx
		with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
			deconv_filter = tf.get_variable(name=name,initializer=tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
		
		deconv = tf.map_fn(lambda x: tf.nn.conv2d_transpose(x[0], deconv_filter, tf.shape(x[1]), strides=[1,pool_size,pool_size,1]), (x1,x2), dtype=tf.float32)
		deconv_output = tf.concat([deconv, x2], -1)
		deconv_output.set_shape([None, None,None,None,output_channels*2])

	return deconv_output

## https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
def adain(content, style, epsilon=1e-5, data_format='channels_first'):
	#axes = [2,3] if data_format == 'channels_first' else [1,2]

	c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
	s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
	c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

	return s_std * (content - c_mean) / c_std + s_mean

def encode_block(inputs, dims, activation_fn, block_idx, max_pool=True, normalizer_fn=None,  module_name="original", ksize1=3, ksize2=3, use_center=True):
	if module_name != "original":
		module_name = "_"+module_name
	else:
		module_name = ""


	conv = tf.map_fn(lambda x: slim.conv2d(x,dims,[ksize1,ksize1], rate=1, normalizer_fn=normalizer_fn, activation_fn=activation_fn,scope='g_conv%d_1%s'%(block_idx, module_name), reuse=tf.AUTO_REUSE), inputs)
	encs = tf.map_fn(lambda x: slim.conv2d(x,dims,[ksize2,ksize2], rate=1, normalizer_fn=normalizer_fn, activation_fn=activation_fn,scope='g_conv%d_2%s'%(block_idx, module_name), reuse=tf.AUTO_REUSE), conv)  

	# global_pool = tf.reduce_max(encs, 0)
	if block_idx > 1:
		global_pool = tf.reduce_max(encs, 0)
	else:
		if use_center == True:
			sh = tf.shape(encs)
			global_pool = encs[sh[0]//2]
		else:
			global_pool = encs[0]

	results = encs
	if max_pool == True:
		results = tf.map_fn(lambda x: slim.max_pool2d(x, [2, 2], padding='SAME'), results)
 
	return results, encs, global_pool

def decode_block(inputs, inputs_early, out_channels, in_channels, activation_fn, block_idx, module_name, ksize=3):
	if module_name != "original":
		module_name = "_"+module_name
	else:
		module_name = ""

	up = upsample_and_concat(inputs, inputs_early, out_channels, in_channels, block_idx=block_idx-6)
	conv = tf.map_fn(lambda x: slim.conv2d(x, out_channels, [ksize,ksize], rate=1, activation_fn=activation_fn, scope='g_conv%d_1%s'%(block_idx,module_name), reuse=tf.AUTO_REUSE), up)
	conv = tf.map_fn(lambda x: slim.conv2d(x, out_channels, [ksize,ksize], rate=1, activation_fn=activation_fn, scope='g_conv%d_2%s'%(block_idx,module_name), reuse=tf.AUTO_REUSE), conv)
	conv.set_shape([None, None, None, None, out_channels])
	global_pool = tf.reduce_max(conv, 0)

	return conv, global_pool

## Coarse module.
def coarse_net(inputs, dims=32, out_channels=4, module_name="original", out_raw=True):
	nets = inputs
	pool1s, conv1s, conv1 = encode_block(nets, dims,  activation_fn=lrelu, module_name=module_name, block_idx=1)
	pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, module_name=module_name, block_idx=2)
	pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, module_name=module_name, block_idx=3)
	pool4s, conv4s, conv4 = encode_block(pool3s, dims*8,  activation_fn=lrelu, module_name=module_name, block_idx=4)
	conv5s, conv5s, conv5 = encode_block(pool4s, dims*16,  activation_fn=lrelu, module_name=module_name, block_idx=5, max_pool=False)

	conv6s, conv6 = decode_block(conv5s, conv4s, dims*8, dims*16, activation_fn=lrelu,  block_idx=6, module_name=module_name)
	conv7s, conv7 = decode_block(conv6s, conv3s, dims*4, dims*8, activation_fn=lrelu, block_idx=7, module_name=module_name)
	conv8s, conv8 = decode_block(conv7s, conv2s, dims*2, dims*4, activation_fn=lrelu, block_idx=8, module_name=module_name)
	conv9s, conv9 = decode_block(conv8s, conv1s, dims, dims*2, activation_fn=lrelu, block_idx=9, module_name=module_name)
	
	nets = conv9s
	if out_raw == True:
		coarse_outs = inputs + tf.map_fn(lambda x: slim.conv2d(x, out_channels, [1,1], rate=1, activation_fn=None, scope='g_conv10', reuse=tf.AUTO_REUSE), nets)
	else:
	   coarse_outs = tf.map_fn(lambda x: slim.conv2d(x, out_channels, [1,1], rate=1, activation_fn=None, scope='g_conv10', reuse=tf.AUTO_REUSE), nets)
	   coarse_outs = tf.map_fn(lambda x: tf.depth_to_space(x, int(np.sqrt(out_channels/3))), coarse_outs)

	return coarse_outs


## Fine module.
def fine_net(inputs, coarse_outs, out_channels=12, dims=32, normalizer_fn=None, module_name="fine"):
	coarse_outs = tf.map_fn(lambda x: dbputils.tf_upsample(x, s=2), coarse_outs)
	res =  inputs - coarse_outs
	inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)


	pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1)
	pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=2)
	pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=3)
	pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=4)
	conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=5, max_pool=False)


	up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
	conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv6_1_%s'%module_name,  reuse=tf.AUTO_REUSE)
	conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
	conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
	conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
	conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, normalizer_fn=normalizer_fn,activation_fn=lrelu, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, normalizer_fn=normalizer_fn, activation_fn=lrelu, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)


	conv10 = slim.conv2d(conv9, out_channels, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
	out = tf.depth_to_space(conv10, int(np.sqrt(out_channels/3)))

	return out

def fine_res_net(inputs, coarse_outs, out_channels=12, dims=32, nres_block=16, normalizer_fn=None, module_name="fine", demosaic=True, use_center=True, use_noise_map=True):
	coarse_outs = tf.map_fn(lambda x: dbputils.tf_upsample(x, s=2), coarse_outs)
	res =  inputs - coarse_outs
	inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)
	if use_noise_map == False:
		inputs_ = tf.concat([inputs, coarse_outs], axis=4)

	pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)
	pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=2, use_center=use_center)
	pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=3, use_center=use_center)
	pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=4, use_center=use_center)
	conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=5, max_pool=False, use_center=use_center)

	net = conv5
	for i in range(nres_block):
		temp = net
		net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu, normalizer_fn=instance_norm, scope='g_res%d_conv1_%s'%(i, module_name))
		net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, normalizer_fn=instance_norm, scope='g_res%d_conv2_%s'%(i, module_name))
		net = se_block(net, dims*16, block_idx=i)
		net = net + temp

	net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, scope='g_res_%s'%module_name)
	net = se_block(net, dims*16, block_idx=nres_block)
	conv5 = net + conv5

	up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
	conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv6_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
	conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
	conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
	conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)


	if demosaic == True:
		conv10 = slim.conv2d(conv9, out_channels, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
		out = tf.depth_to_space(conv10, int(np.sqrt(out_channels/3)))
	else:
		out = slim.conv2d(conv9, out_channels, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)

	return out

def fine_res_net_flow(inputs, coarse_outs, flows, out_channels=12, dims=32, nres_block=16, normalizer_fn=None, module_name="fine", demosaic=True):
	coarse_outs = tf.map_fn(lambda x: dbputils.tf_upsample(x, s=2), coarse_outs)
	res =  inputs - coarse_outs
	inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)

	pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1)

	center = tf.shape(pool1s)[0]//2
	#center = 0
	ref = pool1s[center:center+1, :,:,:,:]
	pool1s = tf.concat([pool1s[:center,:,:,:,:], pool1s[center+1:,:,:,:,:]], axis=0)
	x = tf.concat([pool1s,flows],axis=4)
	pool1s = tf.map_fn(lambda x: tf.contrib.image.dense_image_warp(x[:,:,:,:32], x[:,:,:,32:]), x)

	#pool1s = tf.concat([ref,pool1s], axis=0)
	temp = tf.concat([pool1s[:center,:,:,:,:],ref], axis=0)
	pool1s = tf.concat([temp,pool1s[center:,:,:,:,:]], axis=0)

	pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=2)
	pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=3)
	pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=4)
	conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=5, max_pool=False)
	
	net = conv5
	for i in range(nres_block):
		temp = net
		net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu, normalizer_fn=instance_norm, scope='g_res%d_conv1_%s'%(i, module_name))
		net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, normalizer_fn=instance_norm, scope='g_res%d_conv2_%s'%(i, module_name))
		net = se_block(net, dims*16, block_idx=i)
		net = net + temp

	net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, scope='g_res_%s'%module_name)
	net = se_block(net, dims*16, block_idx=nres_block)
	conv5 = net + conv5

	up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
	conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv6_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
	conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
	conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)

	up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
	conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
	conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=normalizer_fn, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)


	if demosaic == True:
		conv10 = slim.conv2d(conv9, out_channels, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
		out = tf.depth_to_space(conv10, int(np.sqrt(out_channels/3)))
	else:
		out = slim.conv2d(conv9, out_channels, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)

	return out

def se_block(x, dims, block_idx, ratio=16):
	squeeze = tf.reduce_mean(x, axis=[1,2]) ## global avg pool.
	excitation = slim.fully_connected(squeeze, dims//ratio, scope='g_conv_fine_%d_se_fc_1'%block_idx, reuse=tf.AUTO_REUSE)
	excitation = tf.nn.relu(excitation)
	excitation = slim.fully_connected(excitation, dims, scope='g_conv_fine_%d_se_fc_2'%block_idx, reuse=tf.AUTO_REUSE)
	excitation = tf.sigmoid(excitation)
	excitation = tf.reshape(excitation, [-1,1,1,dims])

	#out = x*(0.5+excitation)
	out = x*excitation
	return out