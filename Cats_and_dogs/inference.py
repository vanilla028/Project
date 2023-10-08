import os
import torch
from torchvision import models, transforms
from PIL import Image

# 모델 가중치 파일 경로
model_path = "./models/best.pt"

# 입력 이미지 파일 경로
image_path = "입력_이미지_파일_경로.jpg"

# 클래스 레이블
class_labels = ["cat", "dog"]

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리를 위한 변환기 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 모델 생성 (ResNet-50)
model = models.resnet50(pretrained=False)  # pretrained=False로 설정하여 학습된 가중치 불러오지 않음
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # 클래스 개수에 맞게 마지막 레이어 수정
model.to(device)

# 모델 가중치 불러오기
model.load_state_dict(torch.load(model_path))
model.eval()  # 평가 모드로 전환

# 이미지 로드 및 전처리
image = Image.open(image_path)
image = preprocess(image)
image = image.unsqueeze(0)  # 배치 차원 추가

# GPU를 사용하는 경우
if torch.cuda.is_available():
    image = image.to('cuda')
    model.to('cuda')

# 추론 수행
with torch.no_grad():
    output = model(image)

# 결과 해석
output_prob = torch.nn.functional.softmax(output[0], dim=0)
predicted_class_index = torch.argmax(output_prob).item()
predicted_class = class_labels[predicted_class_index]

print(f"Predicted class: {predicted_class}")
