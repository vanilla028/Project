import torch
from torchvision import transforms

# 모델 로드
model = torch.load("resnet18_model.pt")

# 이미지 전처리 및 준비
image_path = 'test_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# GPU 사용 여부 확인 및 모델 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_image = input_image.to(device)

# 모델 추론
model.eval()
with torch.no_grad():
    outputs = model(input_image)

# 예측 결과 확인
_, predicted_idx = torch.max(outputs, 1)
predicted_label = predicted_idx.item()
print(f"Predicted Label: {predicted_label}")
