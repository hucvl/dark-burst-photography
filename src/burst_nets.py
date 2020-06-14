# set_nets.py: Ahmet Serdar Karadeniz
# description: Networks for set based model.

from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
import numpy as np
from tensorflow.contrib.layers import layer_norm
import math
import utils

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


def encode_block(inputs, dims, activation_fn,  block_idx, max_pool=True,  module_name="original", ksize1=3, ksize2=3):
    if module_name != "original":
        module_name = "_"+module_name
    else:
        module_name = ""


    conv = tf.map_fn(lambda x: slim.conv2d(x,dims,[ksize1,ksize1], rate=1, activation_fn=activation_fn,scope='g_conv%d_1%s'%(block_idx, module_name), reuse=tf.AUTO_REUSE), inputs)
    encs = tf.map_fn(lambda x: slim.conv2d(x,dims,[ksize2,ksize2], rate=1, activation_fn=activation_fn,scope='g_conv%d_2%s'%(block_idx, module_name), reuse=tf.AUTO_REUSE), conv)  
    
    ## se block.
    # if "fine" in module_name:
    #     encs = tf.map_fn(lambda x: se_block(x, dims), encs)

    global_pool = tf.reduce_max(encs, 0)

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

def se_block(x, dims,  scope,  ratio=16):
    squeeze = tf.reduce_mean(x, axis=[1,2]) ## global avg pool.
    excitation = slim.fully_connected(squeeze, dims//ratio, scope=scope)
    excitation = tf.nn.relu(excitation)
    excitation = slim.fully_connected(excitation, dims, scope=scope)
    excitation = tf.sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,dims])

    out = x*excitation
    return out

# def ca_block(x, dims, scope, ratio=16):
#     kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
#     bias_initializer = tf.constant_initializer(value=0.0)

#     squeeze = tf.reduce_mean(x, axis=[1,2], keepdims=True) ## global avg pool.
#     excitation = slim.fully_connected(squeeze, dims//ratio, scope="%s_1"%scope,weights_initializer=kernel_initializer, biases_initializer=bias_initializer)
#     excitation = tf.nn.relu(excitation)
#     excitation = slim.fully_connected(excitation, dims, scope="%s_2"%scope, weights_initializer=kernel_initializer, biases_initializer=bias_initializer)
#     excitation = tf.sigmoid(excitation)
#     #excitation = tf.reshape(excitation, [-1,1,1,dims])

#     out = x*excitation
#     return out


## Coarse module.
def coarse_net(inputs, dims=32, module_name="original"):
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
    coarse_outs = inputs + tf.map_fn(lambda x: slim.conv2d(x, 4, [1,1], rate=1, activation_fn=None, scope='g_conv10', reuse=tf.AUTO_REUSE), nets)
    #coarse_outs = tf.map_fn(lambda x: slim.conv2d(x, 4, [1,1], rate=1, activation_fn=None, scope='g_conv10', reuse=tf.AUTO_REUSE), nets)


    return coarse_outs

## Fine module.
def fine_net(inputs, coarse_outs, dims=32, module_name="fine"):
    coarse_outs = tf.map_fn(lambda x: utils.tf_upsample(x), coarse_outs)
    #res = tf.abs(inputs - coarse_outs) + 0.5
    res =  inputs - coarse_outs
    inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)

    pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, module_name=module_name, block_idx=1)
    pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, module_name=module_name, block_idx=2)
    pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, module_name=module_name, block_idx=3)
    pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, module_name=module_name, block_idx=4)
    conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, module_name=module_name, block_idx=5, max_pool=False)

    up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
    conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1_%s'%module_name,  reuse=tf.AUTO_REUSE)
    conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv6 = se_block(conv6,dims*8)

    up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
    conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv7 = se_block(conv7,dims*4)

    up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
    conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv8 = se_block(conv8,dims*2)

    up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
    conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv9 = se_block(conv9,dims,ratio=8)

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
    out = tf.depth_to_space(conv10, 2)

    #decoder_layers.append(conv10s)
    return out

## Fine module with residual blocks.
def fine_res_net(inputs, coarse_outs, dims=32, module_name="fine"):
    coarse_outs = tf.map_fn(lambda x: utils.tf_upsample(x), coarse_outs)
    #res = tf.abs(inputs - coarse_outs) + 0.5
    res =  inputs - coarse_outs
    inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)

    pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, module_name=module_name, block_idx=1)
    pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, module_name=module_name, block_idx=2)
    pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, module_name=module_name, block_idx=3)
    pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, module_name=module_name, block_idx=4)
    conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, module_name=module_name, block_idx=5, max_pool=False)

    net = conv5
    for i in range(4):
        temp = net
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu, normalizer_fn=layer_norm, scope='g_conv_%s_res_%d_1' % (module_name, i))
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, normalizer_fn=layer_norm, scope='g_conv_%s_res_%d_2' % (module_name, i))
        net = net + temp
      
    net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_conv_%s_res'%module_name)
    conv5 = net + conv5

    up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
    conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1_%s'%module_name,  reuse=tf.AUTO_REUSE)
    conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv6 = se_block(conv6,dims*8)

    up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
    conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv7 = se_block(conv7,dims*4)

    up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
    conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv8 = se_block(conv8,dims*2)

    up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
    conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv9 = se_block(conv9,dims,ratio=8)

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
    out = tf.depth_to_space(conv10, 2)

    #decoder_layers.append(conv10s)
    return out

## Fine module with residual blocks.
def fine_res_se_net(inputs, coarse_outs, dims=32, module_name="fine"):
    coarse_outs = tf.map_fn(lambda x: utils.tf_upsample(x), coarse_outs)
    #res = tf.abs(inputs - coarse_outs) + 0.5
    res =  inputs - coarse_outs
    inputs_ = tf.concat([inputs, res, coarse_outs], axis=4)

    pool1s, conv1s, conv1 = encode_block(inputs_, dims,  activation_fn=lrelu, module_name=module_name, block_idx=1)
    pool2s, conv2s, conv2 = encode_block(pool1s, dims*2, activation_fn=lrelu, module_name=module_name, block_idx=2)
    pool3s, conv3s, conv3 = encode_block(pool2s, dims*4, activation_fn=lrelu, module_name=module_name, block_idx=3)
    pool4s, conv4s, conv4 = encode_block(pool3s, dims*8, activation_fn=lrelu, module_name=module_name, block_idx=4)
    conv5s, conv5s, conv5 = encode_block(pool4s, dims*16, activation_fn=lrelu, module_name=module_name, block_idx=5, max_pool=False)

    net = conv5
    for i in range(4):
        temp = net
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu, normalizer_fn=layer_norm, scope='g_conv_%s_res_%d_1' % (module_name, i))
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=None, normalizer_fn=layer_norm, scope='g_conv_%s_res_%d_2' % (module_name, i))
        net = net + temp 
        net = se_block(net,  dims*16,  scope="g_conv_%s_res_se_%d"%(module_name,i), ratio=16)
      
    net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_conv_%s_res'%module_name)
    conv5 = net + conv5

    up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
    conv6 = slim.conv2d(up6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1_%s'%module_name,  reuse=tf.AUTO_REUSE)
    conv6 = slim.conv2d(conv6, dims*8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv6 = se_block(conv6,dims*8)

    up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
    conv7 = slim.conv2d(up7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv7 = slim.conv2d(conv7, dims*4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv7 = se_block(conv7,dims*4)

    up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
    conv8 = slim.conv2d(up8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv8 = slim.conv2d(conv8, dims*2, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv8 = se_block(conv8,dims*2)

    up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
    conv9 = slim.conv2d(up9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1_%s'%module_name, reuse=tf.AUTO_REUSE)
    conv9 = slim.conv2d(conv9, dims, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2_%s'%module_name, reuse=tf.AUTO_REUSE)
    #conv9 = se_block(conv9,dims,ratio=8)

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10_%s'%module_name, reuse=tf.AUTO_REUSE)
    out = tf.depth_to_space(conv10, 2)

    #decoder_layers.append(conv10s)
    return out