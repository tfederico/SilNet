# Import modules for data interaction and visualisation

import random
import numpy as np
import pandas as pd

# Import modules for image visualisation

from PIL import Image

# Import PyTorch modules

import torch
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader

# Import custom modules

from utils_silnet import reverse_transform, save_model, masks_to_colorimg, plot_side_by_side
from synthhands_dataset import SynthHandsDataset
from train_silnet import train_model, get_optimizer, get_lr_scheduler
from silnet import SilNet

# Parameters

config_path = "config_silnet.csv"
config = pd.read_csv(config_path, header=0, dtype=str)
cpu = config["cpu"].iloc[0]
gpu = config["gpu"].iloc[0]
results_prefix = config["results_prefix"].iloc[0]
images_ext = config["images_ext"].iloc[0]
phases = ["train", "val", "test"]
model_path = config["model_path"].iloc[0]
dataset_path = config["dataset_path"].iloc[0]
imgs_dir = config["imgs_dir"].iloc[0]
seed = int(config["seed"].iloc[0])
deterministic = bool(config["deterministic"].iloc[0])
train_ratio = float(config["train_ratio"].iloc[0])
valid_ratio = float(config["valid_ratio"].iloc[0])
test_ratio = float(config["test_ratio"].iloc[0])
batch_size = int(config["batch_size"].iloc[0])
test_batch_size = int(config["test_batch_size"].iloc[0])
lr = float(config["lr"].iloc[0])
step_size = int(config["step_size"].iloc[0])
gamma = float(config["gamma"].iloc[0])
num_epochs = int(config["num_epochs"].iloc[0])
num_test_batches = int(config["num_test_batches"].iloc[0])

# Set seed(s)

"""random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if deterministic:
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True"""

train_set = SynthHandsDataset("data/train")
valid_set = SynthHandsDataset("data/val")
test_set = SynthHandsDataset("data/test")

image_datasets = {
	phases[0]: train_set, phases[1]: valid_set, phases[2]: test_set
}


dataloaders = {
	phases[0]: DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
	phases[1]: DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
	phases[2]: DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
}


device = torch.device(gpu if torch.cuda.is_available() else cpu)
print(device)

model = SilNet().to(device)

# Observe that all parameters are being optimized
optimizer_ft = get_optimizer(model, lr=lr)

exp_lr_scheduler = get_lr_scheduler(optimizer_ft, step_size=step_size, gamma=gamma)

model = train_model(device, model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
save_model(model, model_path, back_to=gpu)

model.eval()   # Set model to the evaluation mode
batch_num = 0
#iterator = iter(dataloaders[phases[2]])
#for i in range(num_test_batches):
for inputs, labels in dataloaders[phases[2]]:
	#inputs, labels = next(iterator)
	inputs = inputs.to(device)
	labels = labels.to(device)


	# Predict
	pred = model(inputs)
	# The loss functions include the sigmoid function.
	pred = torch.sigmoid(pred)
	pred = pred.data.cpu()

	# Change channel-order and make 3 channels for matplot
	input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

	# Map each channel (i.e. class) to each color
	target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
	pred_rgb = [masks_to_colorimg(x) for x in pred.numpy()]

	#plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

	for i in range(len(input_images_rgb)):
		imgs_comb = np.hstack((input_images_rgb[i], target_masks_rgb[i], pred_rgb[i]))
		imgs_comb = Image.fromarray(imgs_comb)
		imgs_comb.save(results_prefix+"batch"+str(batch_num)+"_sample"+str(i)+images_ext)

	batch_num += 1
