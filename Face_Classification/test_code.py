import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models

from custom_dataset import MyCustomData
from train_to_test_utils import set_augmentations, test
import warnings
warnings.filterwarnings('ignore')

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set augmentations
test_transforms = set_augmentations('test')

# Load Dataset
dataset_path = os.path.join('./', 'dataset')

test_dataset = MyCustomData(
    dataset_path, task='test', transforms=test_transforms)

# Set DataLoader
test_loader = DataLoader(test_dataset, batch_size=1,
                         shuffle=False, num_workers=4, pin_memory=True)

if __name__ == '__main__':
    label_quantity = 10
    # Test Model call
    # model = models.shufflenet_v2_x2_0(pretrained=False) -> 모델 아키텍처 호출
    model = models.swin_t(weights="IMAGENET1K_V1")
    # model.fc = nn.Linear(in_features=2048, out_features=label_quantity)
    # -> 모델 마지막 이미지 특징점 및 라벨 추출 부분 수정 out_features= 라벨 갯수
    model.fc = nn.Linear(in_features=2048, out_features=10)

    # -> 모델 호출 하는 부분
    model_path = './SwinTransformer_best.pt'
    torch.load(model_path, map_location=device)
    model.to(device)

    test(test_loader, device, label_quantity)
    #test(test_loader, model, device)


