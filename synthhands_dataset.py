from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
from skimage import io

class SynthHandsDataset(Dataset):
	def __init__(self, csv_file, imgs_dir, csv_config, transform=None):
		self.imgs_names = pd.read_csv(csv_file, header=None, dtype=str).iloc[:,1]
		self.feature_transform = transform["feature"]
		self.target_transform = transform["target"]
		self.imgs_dir = imgs_dir
		config = pd.read_csv(csv_config, header=0, dtype=str)
		self.images_ext = config["images_ext"].iloc[0]
		self.features_prefix = config["features_prefix"].iloc[0]
		self.targets_prefix = config["targets_prefix"].iloc[0]

	def __len__(self):
		return len(self.imgs_names)

	def __getitem__(self, idx):
		
		idx = int(idx) # why??
		feature_img_name = os.path.join(self.imgs_dir, self.features_prefix + self.imgs_names[idx] + self.images_ext)
		target_img_name = os.path.join(self.imgs_dir, self.targets_prefix + self.imgs_names[idx] + self.images_ext)
		feature_image = Image.open(feature_img_name)
		target_image = Image.open(target_img_name)
		#feature_image = np.array(feature_image)
		#target_image = np.array(target_image)
		
		if self.feature_transform:
			feature_image = self.feature_transform(feature_image)
		if self.target_transform:
			target_image = self.target_transform(target_image)

		return [feature_image, target_image]
