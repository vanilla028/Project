import torch
from torch.utils.data import DataLoader
from customdataset import MycustomDataset
import torch.optim as optim
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import  torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from utils import train

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation
train_transform = A.Compose([
    A.Resize(width=224,height=224),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 데이터셋 및 데이터로더 생성
train_dataset = MycustomDataset("./archive/train/", transform=train_transform)
val_dataset = MycustomDataset("./archive/validataion/", transform=val_transform)
test_dataset = MycustomDataset("./archive/test/", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 생성 (ResNet-50)
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 클래스 개수에 맞게 마지막 레이어 수정
model.to(device)

# 손실 함수, 옵티마이저 설정
criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamP(model.parameters(), lr=0.001)
save_dir = "./"

num_epoch = 100

train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
      save_dir, device)