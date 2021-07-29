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

sys.path.insert(0, "../RAFT/")
sys.path.insert(0, "../RAFT/core/")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import torch
from PIL import Image
import argparse

def save_flow(flo, vis_path):
	flo = flo[0].permute(1,2,0).cpu().numpy()

	flo = flow_viz.flow_to_image(flo)
	print(vis_path)
	cv2.imwrite(vis_path, flo)

def avg_green_channels(im):
	result = np.zeros(shape=(im.shape[0], im.shape[1], im.shape[2], im.shape[3]-1), dtype=im.dtype)
	result[:,:,:,0] = im[:,:,:,0]
	#result[:,:,:,1] = (im[:,:,:,1]+im[:,:,:,2])/2
	result[:,:,:,1] = im[:,:,:,1]
	result[:,:,:,2] = im[:,:,:,3]
	return result

def warp_frames(frames):
	print(frames.shape)
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="restore checkpoint")
	parser.add_argument('--path', help="dataset for evaluation")
	parser.add_argument('--small', action='store_true', help='use small model')
	parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
	parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
	args = parser.parse_args()

	args.model = "../RAFT/models/raft-things.pth"
	args.path = "../RAFT/demo-frames"


	device = torch.device("cuda")
	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model))
	

	model = model.module
	model.to(device)
	model.eval()

	if not os.path.isdir("./results/flow_results"):
		os.makedirs("./results/flow_results")

	center = frames.shape[0] // 2

	warped_frames = []
	with torch.no_grad():
		ref = frames[center,:,:,:,:]
		cv2.imwrite("./results/flow_results/frame_%d.png"%center, np.uint8(avg_green_channels(ref)[0]*255))

		raft_input = np.uint8(avg_green_channels(ref)*255)
		ref_ = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
		print(ref_.shape)
		for i in range(frames.shape[0]):
			if i == center:
				continue
			frame = frames[i, :, :, :, :]
			cv2.imwrite("./results/flow_results/frame_%d.png"%i, np.uint8(avg_green_channels(frame)[0]*255))

			raft_input = np.uint8(avg_green_channels(frame)*255)
			img = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
			padder = InputPadder(ref_.shape)
			ref, img = padder.pad(ref_, img)
			print(ref.shape, img.shape)
			print(ref.min(), ref.max(), img.min(), img.max())
			flow_low, flow_up = model(ref, img, iters=20, test_mode=True)

			flow_up = padder.unpad(flow_up)
			save_flow(flow_up,"./results/flow_results/flow_%d.jpg"%i)
			flo_up = flow_up[0].permute(1,2,0).cpu().numpy()
			flo_low = flow_low[0].permute(1,2,0).cpu().numpy()

			h = flo_up.shape[0]
			w = flo_up.shape[1]
			flo_up = -flo_up
			flo_up[:,:,0] += np.arange(w)
			flo_up[:,:,1] += np.arange(h)[:,np.newaxis]
			warped_frame = cv2.remap(frame[0,:,:,:], flo_up, None, cv2.INTER_LINEAR)
			print(warped_frame.shape)
			cv2.imwrite("./results/flow_results/frame_%d_warped.png"%i, np.uint8(avg_green_channels(np.expand_dims(warped_frame,0))[0]*255))

			warped_frames.append(frame)
			
	print(frames.shape)
	warped_frames.insert(center, frames[center,:,:,:,:])
	warped_frames = np.array(warped_frames)
	print(warped_frames.shape)

	return warped_frames

def warp_frames_with_masking(frames, coarse_frames, result_dir, frame_idx=0):
	print(frames.shape)
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="restore checkpoint")
	parser.add_argument('--path', help="dataset for evaluation")
	parser.add_argument('--small', action='store_true', help='use small model')
	parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
	parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
	args = parser.parse_args()

	args.model = "../RAFT/models/raft-things.pth"
	args.path = "../RAFT/demo-frames"
	#args.small = True


	device = torch.device("cuda")
	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model))
	model = model.module
	model.to(device)
	model.eval()

	# device = torch.device("cpu")
	# model = torch.nn.DataParallel(RAFT(args))
	# model.load_state_dict(torch.load(args.model, map_location=device))	
	# model = model.module
	# model.to(device)
	# model.eval()

	if not os.path.isdir("./results/flow_results"):
		os.makedirs("./results/flow_results")

	center = frames.shape[0] // 2
	#center = 0

	warped_frames = []
	warped_coarse_frames = []
	with torch.no_grad():
		ref = coarse_frames[center,:,:,:,:]
		cv2.imwrite(result_dir+"%04d_%d.jpg"%(frame_idx,center), np.uint8(avg_green_channels(ref)[0]*255))

		raft_input = np.uint8(avg_green_channels(ref)*255)
		ref_ = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
		print(ref_.shape)
		for i in range(coarse_frames.shape[0]):
			if i == center:
				continue
			coarse_frame = coarse_frames[i, :, :, :, :]
			#cv2.imwrite(result_dir+"%d_coarse_%d.jpg"%center, np.uint8(avg_green_channels(coarse_frame)[0]*255))

			raft_input = np.uint8(avg_green_channels(coarse_frame)*255)
			img = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
			padder = InputPadder(ref_.shape)
			ref, img = padder.pad(ref_, img)
			print(ref.shape, img.shape)
			print(ref.min(), ref.max(), img.min(), img.max())
			flow_low, flow_up = model(ref, img, iters=20, test_mode=True)

			flow_up = padder.unpad(flow_up)
			save_flow(flow_up, result_dir+"%04d_%d_flow.jpg"%(frame_idx,i))
			flo_up = flow_up[0].permute(1,2,0).cpu().numpy()
			flo_low = flow_low[0].permute(1,2,0).cpu().numpy()

			flo_up = flow_up[0].permute(1,2,0).cpu().numpy()
			mag, ang = cv2.cartToPolar(flo_up[...,0], flo_up[...,1])
			
			thresh = 0.999
			mask = np.zeros(mag.shape)
			mask[mag > 0] = 1
			mask[mag < thresh] = 0

			kernel_size = 2 * (9 * frames.shape[0]) + 1
			kernel = np.ones((kernel_size,kernel_size), np.uint8)
			mask = cv2.dilate(mask, kernel, iterations=1) 
			resized_mask = cv2.resize(mask, (mask.shape[1]*2, mask.shape[0]*2))

			mask = np.expand_dims(mask, 0)
			mask = np.expand_dims(mask, -1)
			coarse_frame = coarse_frame * (1-mask) + coarse_frames[center,:,:,:,:] * mask

			frame = frames[i, :, :, :, :]
			resized_mask = np.expand_dims(resized_mask, 0)
			resized_mask = np.expand_dims(resized_mask, -1)
			frame = frame * (1-resized_mask) + frames[center, :, :, :, :] * resized_mask

			#cv2.imwrite("./results/flow_results/coarse_frame_%d_warped.png"%i, np.uint8(avg_green_channels(coarse_frame)[0]*255))
			#cv2.imwrite("./results/flow_results/mask_%d.png"%i, np.uint8(resized_mask[0,:,:,0]*255))

			warped_coarse_frames.append(coarse_frame)
			warped_frames.append(frame)
			

	warped_frames.insert(center, frames[center,:,:,:,:])
	warped_frames = np.array(warped_frames)
	warped_coarse_frames.insert(center, coarse_frames[center,:,:,:,:])
	warped_coarse_frames = np.array(warped_coarse_frames)

	return warped_frames, warped_coarse_frames

def compute_flow(frames):
	print(frames.shape)
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="restore checkpoint")
	parser.add_argument('--path', help="dataset for evaluation")
	parser.add_argument('--small', action='store_true', help='use small model')
	parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
	parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
	args = parser.parse_args()

	args.model = "../RAFT/models/raft-things.pth"
	args.path = "../RAFT/demo-frames"


	device = torch.device("cuda")
	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model))
	

	model = model.module
	model.to(device)
	model.eval()

	if not os.path.isdir("./results/flow_results"):
		os.makedirs("./results/flow_results")

	center = frames.shape[0] // 2

	flows = []
	with torch.no_grad():
		ref = frames[center,:,:,:,:]
		cv2.imwrite("./results/flow_results/coarse_frame_%d.png"%center, np.uint8(avg_green_channels(ref)[0]*255))

		raft_input = np.uint8(avg_green_channels(ref)*255*1.5)
		ref_ = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
		print(ref_.shape)
		for i in range(frames.shape[0]):
			if i == center:
				continue
			frame = frames[i, :, :, :, :]
			cv2.imwrite("./results/flow_results/coarse_frame_%d.png"%i, np.uint8(avg_green_channels(frame)[0]*255))

			raft_input = np.uint8(avg_green_channels(frame)*255)
			img = torch.from_numpy(raft_input).permute(0, 3, 1, 2).float().to(device)
			padder = InputPadder(ref_.shape)
			ref, img = padder.pad(ref_, img)
			print(ref.shape, img.shape)
			print(ref.min(), ref.max(), img.min(), img.max())
			flow_low, flow_up = model(img, ref, iters=20, test_mode=True)


			flow_up = padder.unpad(flow_up)


			flo_up = flow_up[0].permute(1,2,0).cpu().numpy()
			mag, ang = cv2.cartToPolar(flo_up[...,0], flo_up[...,1])
			print(mag.max(), mag.min())
			flo_up[mag < 10] = 0
			flow_up = torch.from_numpy(flo_up).permute(2,0,1).unsqueeze(0)

			save_flow(flow_up, "./results/flow_results/flow_%d.jpg"%i)
			flo_up = flow_up[0].permute(1,2,0).cpu().numpy()
			flo_low = flow_low[0].permute(1,2,0).cpu().numpy()

			flows.append(np.expand_dims(-flo_up, 0))
			
			h = flo_up.shape[0]
			w = flo_up.shape[1]
			flo_up = -flo_up
			flo_up[:,:,0] += np.arange(w)
			flo_up[:,:,1] += np.arange(h)[:,np.newaxis]
			warped_coarse = cv2.remap(frame[0,:,:,:], flo_up, None, cv2.INTER_LINEAR)
			print(warped_coarse.shape)
			cv2.imwrite("./results/flow_results/coarse_frame_%d_warped.png"%i, np.uint8(avg_green_channels(np.expand_dims(warped_coarse,0))[0]*255))
			
	flows = np.array(flows)
	print(flows.shape)

	return flows