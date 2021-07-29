# This code is revised from voxel flow by Ziwei Liu https://github.com/liuziwei7/voxel-flow

from __future__ import print_function
import glob
import numpy as np
import os
#import Queue
import queue as Queue
import random
import threading
import cv2
import pdb

data_dir = "/userfiles/hpc-skaradeniz/DRV/"


def load_fn_example(train_id):
    print(train_id)
    gt_path = glob.glob(data_dir + '/long/%s/half0001*.png'%train_id)[0]
    #pdb.set_trace()
    im = cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
    gt_im = np.expand_dims(np.float32(im/65535.0),axis = 0)
    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/%s/*.png'%train_id))
    #choose two random frames from the same video
    ind_seq = np.random.random_integers(0,len(in_files)-2)
    in_path = in_files[ind_seq]
    im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
    in_im1 = np.expand_dims(np.float32(im/65535.0),axis = 0)
    ind_seq2 = np.random.random_integers(0,len(in_files)-2)
    if ind_seq2 == ind_seq:
        ind_seq2 += 1
    in_path = in_files[ind_seq2]
    im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
    in_im2 = np.expand_dims(np.float32(im/65535.0),axis = 0)
    in_im = np.concatenate([in_im1,in_im2],axis=0)
    return (in_im, gt_im)

def load_fn_burst(train_id, n_frames=8, ps=918):
    #print(train_id)
    #gt_path = glob.glob(data_dir + '/long/%s/0001*.png'%train_id)[0]
    gt_path = glob.glob(data_dir + '/long/%s/half0001*.png'%train_id)[0]
    #pdb.set_trace()
    im = cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gt_im = np.expand_dims(np.float32(im/65535.0),axis = 0)
    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/%s/*.png'%train_id))
    
    #choose n random frames from the same video
    indices = np.random.choice(len(in_files), n_frames, replace=False)
    #print("Choosen frames: ", indices)

    # crop
    H = gt_im.shape[1]
    W = gt_im.shape[2]

    # xx = np.random.randint(0, W//2 - ps)
    # yy = np.random.randint(0, H//2 - ps)
    #gt_im = gt_im[:, yy*2:yy*2 + ps*2, xx*2:xx*2 + ps*2, :]

    xx = np.random.randint(0, W - ps + 1)
    yy = np.random.randint(0, H - ps + 1)
    gt_im = gt_im[:, yy:yy + ps, xx:xx+ ps, :]

    r1 = np.random.randint(2,size=1)[0]
    r2 = np.random.randint(2,size=1)[0]
    r3 = np.random.randint(2,size=1)[0]

    if r1 == 1:  # random flip
        gt_im = np.flip(gt_im, axis=1)
    if r2 == 1:
        gt_im = np.flip(gt_im, axis=2)
    if r3 == 1:  # random transpose
        gt_im = np.transpose(gt_im, (0,2,1,3)) 

    inputs = []
    inputs_low = []
    for i in range(n_frames):
        in_path = in_files[indices[i]]
        im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
        im_replicated = np.zeros(shape=(im.shape[0],im.shape[1], im.shape[2]+1),dtype=im.dtype)
        im_replicated[:,:,0] = im[:,:,2]
        im_replicated[:,:,1] = im[:,:,1]
        im_replicated[:,:,2] = im[:,:,1]
        im_replicated[:,:,3] = im[:,:,0]

        im_replicated = im_replicated[yy:yy + ps, xx:xx + ps, :]
      
        if r1 == 1:  # random flip
            im_replicated = np.flip(im_replicated, axis=0)
        if r2 == 1:
            im_replicated = np.flip(im_replicated, axis=1)
        if r3 == 1:  # random transpose
            im_replicated = np.transpose(im_replicated, (1,0,2))

        im_replicated = np.float32(im_replicated/65535.0)
        im_low = cv2.resize(im_replicated, (im_replicated.shape[1]//2, im_replicated.shape[0]//2))
        
        in_np = np.expand_dims(im_replicated,axis = 0)
        in_np_low = np.expand_dims(im_low, 0)

        inputs.append(in_np)
        inputs_low.append(in_np_low)
    
    
    inputs = np.array(inputs)
    inputs_low = np.array(inputs_low)
    return (inputs, inputs_low, gt_im)


class DummpyData(object):
    def __init__(self, data):
        self.data = data
    def __cmp__(self, other):
        return 0
    def __lt__(self, other):
        return 0
    def __le__(self,other):
        return 0

def prefetch_job(load_fn, prefetch_queue, data_list, shuffle, prefetch_size):
    self_data_list = np.copy(data_list)
    data_count = 0
    total_count = len(self_data_list)
    idx = 0
    while True:
        if shuffle:
            if data_count == 0:
                random.shuffle(self_data_list)

            data = load_fn(self_data_list[data_count]) #Load your data here.

            idx = random.randint(0, prefetch_size)
            dummy_data = DummpyData(data)

            prefetch_queue.put((idx, dummy_data), block=True)

        data_count = (data_count + 1) % total_count

class PrefetchQueue(object):
    def __init__(self, load_fn, data_list, batch_size=32, prefetch_size=None, shuffle=True, num_workers=4):
        self.data_list = data_list
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size
        self.load_fn = load_fn
        self.batch_size = batch_size
        if prefetch_size is None:
            self.prefetch_size = 4 * batch_size

        # Start prefetching thread
        # self.prefetch_queue = Queue.Queue(maxsize=prefetch_size)
        self.prefetch_queue = Queue.PriorityQueue(maxsize=prefetch_size)
        for k in range(num_workers):
            t = threading.Thread(target=prefetch_job,
            args=(self.load_fn, self.prefetch_queue, self.data_list,
                  self.shuffle, self.prefetch_size))
            t.daemon = True
            t.start()

    def get_batch(self):
        data_list = []
        #for k in range(0, self.batch_size):
          # if self.prefetch_queue.empty():
          #   print('Prefetch Queue is empty, waiting for data to be read.')
        _, data_dummy = self.prefetch_queue.get(block=True)
        data = data_dummy.data
          #data_list.append(data)
        return data
