import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
import torch


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'val', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = data_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'clear')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		# source_img = read_img(os.path.join(self.root_dir, 'cloud', img_name)) * 2 - 1
		# target_img = read_img(os.path.join(self.root_dir, 'clear', img_name)) * 2 - 1
		# source_img=source_img /255
		# target_img=target_img/255


		# if self.mode == 'train':
		# 	[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		# if self.mode == 'valid':
		# 	[source_img, target_img] = align([source_img, target_img], self.size)

		# return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}
		clear = read_img(os.path.join(self.root_dir, 'clear', img_name))
		haze =read_img(os.path.join(self.root_dir, 'cloud', img_name))
		clear = clear / 255
		haze = haze / 255
		## 除以最大值
		# clear = clear / clear.max()
		# haze =haze / haze.max()
		## 最大最小归一化
		# clear=(clear-clear.min())/(clear.max()-clear.min())
		# haze = (haze - haze.min()) / (haze.max()- haze.min())
		
		if self.mode=="train":
			haze, clear = self.augment([haze, clear], 512, 0, True)
		else:
			haze, clear = self.align([haze, clear], 512)

		clearCopy=clear.copy()
		hazeCopy=haze.copy()
		clear = torch.from_numpy(clearCopy).to(torch.float32)
		haze = torch.from_numpy(hazeCopy).to(torch.float32)

		# 源码要求将[0,1]转换到[-1,1]之间，所用方法如下
		clear = clear * 2 - 1
		haze = haze * 2 - 1

		return {'source': haze, 'target': clear, 'filename':os.path.join(self.root_dir, 'cloud', img_name)}
	
	def augment(self, imgs, size=256, edge_decay=0., only_h_flip=False):
		_, H, W = imgs[0].shape
		Hc, Wc = [size, size]

		# simple re-weight for the edge
		if random.random() < Hc / H * edge_decay:
			Hs = 0 if random.randint(0, 1) == 0 else H - Hc
		else:
			Hs = random.randint(0, H - Hc)

		if random.random() < Wc / W * edge_decay:
			Ws = 0 if random.randint(0, 1) == 0 else W - Wc
		else:
			Ws = random.randint(0, W - Wc)

		for i in range(len(imgs)):
			imgs[i] = imgs[i][:, Hs:(Hs + Hc), Ws:(Ws + Wc)]

		# horizontal flip
		if random.randint(0, 1) == 1:
			for i in range(len(imgs)):
				imgs[i] = np.flip(imgs[i], axis=1)

		if not only_h_flip:
			# bad data augmentations for outdoor
			rot_deg = random.randint(0, 3)
			for i in range(len(imgs)):
				imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

		return imgs

	def align(self, imgs, size=256):
		_, H, W = imgs[0].shape
		Hc, Wc = [size, size]

		Hs = (H - Hc) // 2
		Ws = (W - Wc) // 2
		for i in range(len(imgs)):
			imgs[i] = imgs[i][:, Hs:(Hs + Hc), Ws:(Ws + Wc)]

		return imgs

class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
