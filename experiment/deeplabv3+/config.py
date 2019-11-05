# ----------------------------------------

# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
		self.EXP_NAME = 'deeplabv3+'

		self.DATA_NAME = 'auto'
		self.DATA_AUG = True
		self.DATA_WORKERS = 8
		self.DATA_RESCALE = 512
		self.DATA_RANDOMCROP = 512
		self.DATA_RANDOMROTATION = 0
		self.DATA_RANDOMSCALE = 3
		self.DATA_RANDOM_H = 0
		self.DATA_RANDOM_S = 0
		self.DATA_RANDOM_V = 0
		self.DATA_RANDOMFLIP = 0
		
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'xception'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 35
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model')

		self.TRAIN_LR = 0.02
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00004
		self.TRAIN_POWER = 0.9
		self.TRAIN_GPUS = 4
		self.TRAIN_BATCHES = 16
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 240
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True
		self.TRAIN_BN_MOM = 0.1	
		self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'model','xception_pytorch_imagenet.pth') #pretrain model download url: https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.TEST_GPUS = 4
		self.TEST_BATCHES = 48
		self.TEST_MULTISCALE = [0.5,1.0,1.5]
		self.TEST_FLIP = False
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'model','mymodel.pth') #your model here
		
		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)



cfg = Configuration() 	
