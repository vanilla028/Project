from customdataset import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import medels
import torch

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# train aug
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])
# val aug
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# 데이터셋 생성
train_dataset = custom_dataset("./data/train", transform=train_transform)
val_dataset = custom_dataset("./data/val", transform=val_transform)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

# 모델 정의
model = models.resnet50(pretrained=True)
num_classes = len(train_dataset.label_dict)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 모델 저장 경로
model_path = "./saved_models/resnet50_model.pt"

# 손실 함수 정의
criterion = torch.nn.CrossEntropyLoss()

# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 훈련 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 검증 데이터에 대한 성능 평가
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    # 모델 저장
    torch.save(model.state_dict(), model_path)

    # 결과 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {(correct/total)*100:.2f}%")
