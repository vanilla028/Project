from customdataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch

# 테스트 데이터셋 구성
test_dataset = CustomDataset("./testdata", transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 모델 불러오기
model = torch.load("resnet18_model.pt")

# 테스트 데이터셋을 사용하여 모델 평가
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = (correct / total) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
