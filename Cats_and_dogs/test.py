import os
import torch
from torch.utils.data import DataLoader
from customdataset import MyCustomDataset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
import torch.optim as optim
import time

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation
test_transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 테스트 데이터셋 및 데이터로더 생성
test_dataset = MyCustomDataset("./archive/test/", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 생성 (ResNet-50)
model = models.resnet50(pretrained=False)  # pretrained=False로 설정하여 학습된 가중치 불러오지 않음
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 클래스 개수에 맞게 마지막 레이어 수정
model.to(device)

# 모델 가중치 불러오기 (학습이 완료된 모델의 가중치)
model_path = "./models/best.pt"  # 모델 가중치 파일 경로
model.load_state_dict(torch.load(model_path))
model.eval()  # 평가 모드로 전환

# 손실 함수 설정 (필요한 경우)
criterion = LabelSmoothingCrossEntropy()

# 테스트 함수 정의
def test(model, test_loader, criterion, device):
    print("Testing........")
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1

    avrg_loss = total_loss / cnt
    test_acc = (correct / total * 100)
    print("Acc >> {:.2f} Average loss >> {:.4f}".format(
        test_acc,
        avrg_loss
    ))

# 모델 테스트
test(model, test_loader, criterion, device)
