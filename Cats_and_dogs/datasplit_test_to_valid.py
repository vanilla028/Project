import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# 원래의 Test 폴더와 빈 Validation 폴더 경로 설정
test_dogs_folder = "./archive/test/dogs/*"
validation_folder = "./archive/validation/"

# Validation 폴더 생성
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(os.path.join(validation_folder, "cats"), exist_ok=True)
os.makedirs(os.path.join(validation_folder, "dogs"), exist_ok=True)

# 원래의 Test 폴더와 빈 Validation 폴더 경로 설정
test_dogs_folder = "./archive/test/dogs"
validation_dogs_folder = "./archive/validation/dogs"

test_cats_folder = "./archive/test/cats"
validation_cats_folder = "./archive/validation/cats"

# Test 폴더의 파일 목록 가져오기
test_dogs_files = glob.glob(os.path.join(test_dogs_folder, "*"))
test_cats_files = glob.glob(os.path.join(test_cats_folder, "*"))

# Test 데이터 중 50%를 Valid 폴더로 이동
validation_dogs_files, remaining_dogs_files = train_test_split(test_dogs_files, test_size=0.5, random_state=42)
validation_cats_files, remaining_cats_files = train_test_split(test_cats_files, test_size=0.5, random_state=42)

# Validation 폴더로 이미지 파일 이동
for file in validation_dogs_files:
    dst_path = os.path.join(validation_dogs_folder, os.path.basename(file))
    shutil.move(file, dst_path)

for file in validation_cats_files:
    dst_path = os.path.join(validation_cats_folder, os.path.basename(file))
    shutil.move(file, dst_path)
