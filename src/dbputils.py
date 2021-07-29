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

def shift_samples(input_images, max_err=4):
    shifted_images = [input_images[0]]
    for i in range(1,len(input_images)):
        img = input_images[i][0,:,:,:]
        error_x = np.random.randint(0, max_err)
        error_y = np.random.randint(0, max_err)
        shifted = np.pad(img, ((max_err//2, max_err//2), (max_err//2,max_err//2), (0, 0)), mode='reflect')
        shifted = shifted[error_y:shifted.shape[0]-(max_err-error_y), error_x:shifted.shape[1]-(max_err-error_x), :]
        shifted_images.append(np.expand_dims(shifted, 0))
        
    return np.array(shifted_images)

def crop_samples(input_images, gt_image, gt_image_raw, ps=512, raw_ratio=2):
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

    gt_patch = gt_image[:, yy*raw_ratio:yy*raw_ratio +ps*raw_ratio, xx*raw_ratio:xx*raw_ratio +ps*raw_ratio, :]
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
        img = np.expand_dims(resize(img[0,:,:,:], r=r), 0)
        input_images_low.append(img)
    return np.array(input_images_low)


def pack_raw(raw, black_level=512, maxpix=16383):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (maxpix - black_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    r = im[0:H:2, 0:W:2, :]
    g1 = im[0:H:2, 1:W:2, :]
    g2 = im[1:H:2, 1:W:2, :]
    b = im[1:H:2, 0:W:2, :]

    out = np.concatenate((r, g1, g2, b), axis=2)
    
    return out

def unpack_raw(raw, packed):
    im = raw.raw_image_visible.astype(np.float32)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = im
    out[0:H:2, 0:W:2] = packed[:,:, 0]
    out[0:H:2, 1:W:2] = packed[:, :, 1]
    out[1:H:2, 1:W:2] = packed[:, :, 2]
    out[1:H:2, 0:W:2] = packed[:, :, 3]
    
    out = np.minimum(out*(16383-512) + 512, 16383)
    return out

def pack_fuji_raw(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6
    #print("Packed img of shape: ", img_shape, "to: ",H,W)

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
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
