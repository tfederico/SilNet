from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from os.path import isfile, join

class SynthHandsDataset(Dataset):
	def __init__(self, imgs_dir, transform=None):
		self.feature_transform = transform["feature"]
		self.target_transform = transform["target"]
		self.imgs_dir = imgs_dir
		self.images_ext = ".png"
		self.features_prefix = "feature"
		self.targets_prefix = "mask"
		self.length = len([f for f in listdir(imgs_dir) if isfile(join(imgs_dir, f))])//2

	def __len__(self):
		return self.length

	def __getitem__(self, idx):

		idx = str(int(idx)) # why??
		feature_img_name = join(self.imgs_dir, self.features_prefix + idx + self.images_ext)
		target_img_name = join(self.imgs_dir, self.targets_prefix + idx + self.images_ext)
		feature_image = Image.open(feature_img_name)
		target_image = Image.open(target_img_name)

		if self.feature_transform:
			feature_image = self.feature_transform(feature_image)
		if self.target_transform:
			target_image = self.target_transform(target_image)

		return [feature_image, target_image]
