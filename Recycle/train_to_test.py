import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
from custom_dataset import CustomDataset
from train_to_test_utils import set_augmentations, train, test
import warnings
warnings.filterwarnings('ignore')


# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set augmentations
train_transforms = set_augmentations('train')
valid_transforms = set_augmentations('valid')

# Load Dataset
dataset_path = os.path.join('../', 'dataset')

train_dataset = CustomDataset(dataset_path, task='train', transforms=train_transforms)
valid_dataset = CustomDataset(dataset_path, task='valid', transforms=valid_transforms)
test_dataset = CustomDataset(dataset_path, task='test', transforms=valid_transforms)

# Set DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
# print(len(train_dataset), '##', len(train_loader))
# exit()

# Call model
label_quantity = 12
model = models.shufflenet_v2_x2_0(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=label_quantity)
model.to(device)
# print(model._get_name())
# exit()

# Set parameters
criterion = LabelSmoothingCrossEntropy()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
epoch_num = 100


if __name__ == '__main__':
    # Train and validate
    train(train_loader, valid_loader, epoch_num, model, device, criterion, optimizer, scheduler)

    # Save last.pt
    torch.save(model.state_dict(), './last.pt')

    test(test_loader, device, label_quantity)
