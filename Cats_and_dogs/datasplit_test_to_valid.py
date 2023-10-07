import os
import shutil
from sklearn.model_selection import train_test_split

# 원래의 Test 폴더와 빈 Validation 폴더 경로 설정
test_folder = "./archive/test/"
validation_folder = "./archive/validation/"

# Test 폴더의 파일 목록 가져오기
test_files = os.listdir(test_folder)

# Test 데이터 중 50%를 Valid 폴더로 이동
validation_files, remaining_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Validation 폴더로 이미지 파일 이동
for file in validation_files:
    src_path = os.path.join(test_folder, file)
    dst_path = os.path.join(validation_folder, file)
    shutil.move(src_path, dst_path)

