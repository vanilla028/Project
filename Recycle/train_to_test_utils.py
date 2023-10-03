import os
import sys
import copy

import torch
import torch.nn as nn
from torchvision import models

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def set_augmentations(task):
    if task == 'train':
        temp_transforms = A.Compose([
            A.Resize(width=224, height=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.06, rotate_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=17, g_shift_limit=17, b_shift_limit=17, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomShadow(p=0.5),
            A.RandomFog(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ])
    elif task in ['valid', 'test']:
        temp_transforms = A.Compose([
            A.Resize(width=224, height=224),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ])
    else:
        print('#' * 30, 'Please confirm your task name', '#' * 30)

    return temp_transforms

def visualize_augmentations(dataset, index=0, samples=20, columns=5):
    dataset = copy.deepcopy(dataset)
    dataset.transforms = A.Compose([transform for transform in dataset.transforms if not isinstance(transform, (A.Normalize, ToTensorV2))])

    rows = samples // columns
    _, axis = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 6))

    for i in range(samples):
        image, _ = dataset[index]
        axis.ravel()[i].imshow(image)
        # image = image.permute(1, 2, 0)
        # print('#' * 50, image.shape)
        # https://stackoverflow.com/questions/50963283/imshow-doesnt-need-convert-from-bgr-to-rgb
        # https://github.com/philipperemy/keract/issues/46
        axis.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()


def train(train_loader, valid_loader, epoch_num, model, device, criterion, optimizer, scheduler):
    model_name = model._get_name()
    best_valid_accuracy = 0.0

    train_steps = len(train_loader)
    
    save_path = os.path.join('./', '{}_best.pt'.format(model_name))

    accuracy_df = pd.DataFrame(index=list(range(epoch_num)), columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy'])

    if os.path.exists(save_path):
        best_valid_accuracy = max(pd.read_csv('./{}_accuracy.csv'.format(model_name))['Accuracy'].tolist())

    for epoch in range(1, epoch_num + 1):
        train_running_loss = 0.0
        train_corrects = 0
        train_length = 0

        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')

        for _, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_corrects += (torch.argmax(outputs, dim=1) == labels).sum().item()
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()
            train_length += images.size(0)

            train_bar.desc = 'Train Epoch: [{:3d} / {:3d}]   Loss: {:.3f}'.format(
                epoch, epoch_num, train_loss.data
            )
        
        scheduler.step()

        train_accuracy = train_corrects / train_length
        train_average_loss = train_running_loss / train_steps

        valid_accuracy, valid_average_loss = validate(valid_loader, model, device, criterion)

        accuracy_df.loc[epoch, 'Epoch'] = epoch
        accuracy_df.loc[epoch, 'Train Loss'] = round(train_average_loss, 3)
        accuracy_df.loc[epoch, 'Train Accuracy'] = round(train_accuracy, 3)
        accuracy_df.loc[epoch, 'Valid Loss'] = round(valid_average_loss, 3)
        accuracy_df.loc[epoch, 'Valid Accuracy'] = round(valid_accuracy, 3)

        print('Epoch: [{:3d} / {:3d}]   Train Loss: {:.3f}   Train Accuracy: {:.3f}   Valid Loss: {:.3f}   Valid Accuracy: {:.3f}'.format(
            epoch, epoch_num, train_average_loss, train_accuracy, valid_average_loss, valid_accuracy
        ))
    
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), save_path)
            
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './{}_{}_epoch.pt'.format(model_name, epoch))

        if epoch == epoch_num:
            accuracy_df.to_csv('./{}_accuracy.csv'.format(model_name), index=False)

def validate(valid_loader, model, device, criterion):
    valid_steps = len(valid_loader)
    valid_running_loss = 0.0
    valid_corrects = 0
    valid_length = 0
    
    model.eval()
    with torch.no_grad():
        
        valid_bar = tqdm(valid_loader, file=sys.stdout, colour='red')

        for data in valid_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            valid_loss = criterion(outputs, labels)
            valid_corrects += (torch.argmax(outputs, dim=1) == labels).sum().item()
            valid_running_loss += valid_loss.item()
            valid_length += images.size(0)

    valid_accuracy = valid_corrects / valid_length
    valid_average_loss = valid_running_loss / valid_steps
    
    return valid_accuracy, valid_average_loss


def test(test_loader, device, label_quantity):
    # Model call
    model = models.shufflenet_v2_x2_0(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=label_quantity)
    model.load_state_dict(torch.load('./{}_best.pt'.format(model._get_name()), map_location=device))
    model.to(device)
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            images, labels = image.to(device), label.to(device)

            output = model(images)
            _, argmax = torch.max(output, dim=1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()
        
        accuracy = accuracy_function(correct, total)
        print('Accuracy: ', accuracy)


def accuracy_function(correct, total):
    accuracy = correct / total * 100
    return accuracy


# import os
# from custom_dataset import CustomDataset
# data_path = os.path.join('../', 'dataset')
# transforms = set_augmentations('train')
# train_dataset = CustomDataset(data_path, transforms=transforms, task='train')
# visualize_augmentations(train_dataset)