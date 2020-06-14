import os
import tensorflow as tf
import numpy as np
import cv2


def count_params():
	total_parameters = 0
	for variable in tf.trainable_variables():
    		shape = variable.get_shape()
    		variable_parameters = 1
    		for dim in shape:
        		variable_parameters *= dim.value
    		total_parameters += variable_parameters
	return total_parameters

## https://github.com/google-research/google-research/blob/master/unprocessing/process.py
def gamma_compression(images, gamma=2.2):
  """Converts from linear to gamma space."""
  # Clamps to prevent numerical instability of gradients near zero.
  return tf.maximum(images, 1e-8) ** (1.0 / gamma)


def shuffle_samples(input_images):
    input_images = np.array(input_images)
    indices = np.arange(input_images.shape[0])
    np.random.shuffle(indices)
    input_images = input_images[indices]
    return np.array(input_images)


def augment_samples(input_images, gt_image, gt_image_raw):
    t1, t2, t3 = False, False, False
    if np.random.randint(2, size=1)[0] == 1:  # random flip
        gt_image = np.flip(gt_image, axis=1)
        gt_image_raw = np.flip(gt_image_raw, axis=1)
        t1 = True
    if np.random.randint(2, size=1)[0] == 1:
        gt_image = np.flip(gt_image, axis=2)
        gt_image_raw = np.flip(gt_image_raw, axis=2)
        t2 = True
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        gt_image = np.transpose(gt_image, (0, 2, 1, 3))
        gt_image_raw = np.transpose(gt_image_raw, (0, 2, 1, 3))
        t3 = True

    new_images = []
    for i in range(len(input_images)):
        img = input_images[i]
        if t1 == True:
            img = np.flip(img, axis=1)
        if t2 == True:
            img = np.flip(img, axis=2)
        if t3 == True:
            img = np.transpose(img, (0, 2, 1, 3))
        new_images.append(img)

    return np.array(new_images), gt_image, gt_image_raw

def crop_samples(input_images, gt_image, gt_image_raw, ps=512):
    in_image = input_images[0]
    H = in_image.shape[1]
    W = in_image.shape[2]
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    input_patches = []
    for k in range(len(input_images)):
        input_full = input_images[k]
        input_patch = input_full[:,yy:yy + ps, xx:xx + ps, :]
        input_patches.append(input_patch)

    gt_patch = gt_image[:, yy*2:yy*2 +ps*2, xx*2:xx*2 +ps*2, :]
    gt_raw_patch = gt_image_raw[:, yy:yy + ps, xx:xx + ps, :]
    return np.array(input_patches), gt_patch, gt_raw_patch

def tf_downsample(input, s=2, nn=False):
    sh = tf.shape(input)
    newShape =  sh[1:3] / s
    newShape = tf.cast(newShape, tf.int32)
    if nn == False:
    	output = tf.image.resize_bilinear(input, newShape) 
    else:
    	output = tf.image.resize_nearest_neighbor(input, newShape)
    return output

def tf_upsample(input, s=2, nn=False):
    sh = tf.shape(input)
    newShape = s * sh[1:3]
    if nn == False:
    	output = tf.image.resize_bilinear(input, newShape) 
    else:
    	output = tf.image.resize_nearest_neighbor(input, newShape)
    return output

def resize_samples(input_images, r=2):
    input_images_low = []
    for i in range(len(input_images)):
        img = input_images[i]
        #print(img.shape)
        img = np.expand_dims(resize(img[0,:,:,:], r=r), 0)
        input_images_low.append(img)
    return np.array(input_images_low)


def pack_raw(raw, black_level=512):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    
    return out


def unpack_raw(raw, packed):
    im = raw.raw_image_visible.astype(np.float32)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = im
    out[0:H:2, 0:W:2] =  packed[:,:,0]
    out[0:H:2, 1:W:2] = packed[:, :, 1]
    out[1:H:2, 1:W:2] = packed[:, :, 2]
    out[1:H:2, 0:W:2] = packed[:, :, 3]
    
    out = np.minimum(out*(16383-512) + 512, 16383)
    return out

def to_rawrgb(bayer_arr):
    arr = np.zeros((bayer_arr.shape[0], bayer_arr.shape[1], 3))
    arr[:,:,0] = bayer_arr[:,:,0]
    arr[:,:,1] = (bayer_arr[:,:,1]+bayer_arr[:,:,3])/2
    arr[:,:,2] = bayer_arr[:,:,2]
    return arr

def resize(img, r=2, interpolation=cv2.INTER_LINEAR):
   img = cv2.resize(img, (int(img.shape[1]/r), int(img.shape[0]/r)), interpolation=interpolation)
   return img

def d_set_for_id(d_id=0):
    if d_id == 0:
	    return "train"
    if d_id == 2:
	    return "val"
    else:
	    return "test"

def get_burst_paths(in_path, n_burst):
    paths = []
    complete = True
    for i in range(n_burst):
        path = in_path.replace("_00_","_0%d_"%i)
        if os.path.isfile(path):
            paths.append(path)
        else:
            complete = False
            continue
    return paths, complete
