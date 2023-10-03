from sklearn.model_selection import train_test_split
import glob
import os
import shutil

# 데이터 나누기

"""
data
    - train
        - dekopon
            - dekopon_0000.png
        - grapefruit
        - kanpei
        - orange
    - val
        - dekopon
            - dekopon_0000.png
        - grapefruit
        - kanpei
        - orange
    - test
        - dekopon
            - dekopon_0000.png
        - grapefruit
        - kanpei
        - orange
"""

image_path = "./dataset/image/"


# dekopon 이미지 경로 -> list
dekopon_data = glob.glob(os.path.join(image_path, "dekopon", "*.png"))
grapefruit_data = glob.glob(os.path.join(image_path, "grapefruit", "*.png"))
kanpei_data = glob.glob(os.path.join(image_path, "kanpei", "*.png"))
orange_data = glob.glob(os.path.join(image_path, "orange", "*.png"))

  
# train:val = 9:1
dekopon_data_train, dekopon_data_val = train_test_split(
    dekopon_data, test_size=0.1, random_state=7777)

grapefruit_data_train, grapefruit_data_val = train_test_split(
    grapefruit_data, test_size=0.1, random_state=7777)

kanpei_data_train, kanpei_data_val = train_test_split(
    kanpei_data, test_size=0.1, random_state=7777)

orange_data_train, orange_data_val = train_test_split(
    orange_data, test_size=0.1, random_state=7777)

  #./dataset/image/dekopon/dekopon_105.png
  # 복사는 shutil.copy  
for i in dekopon_data_train:
     file_name = os.path.basename(i)
     os.makedirs("./data/train/dekopon/", exist_ok=True)
     shutil.move(i, f"./data/train/dekopon/{file_name}")

 for i in dekopon_data_val:
     file_name = os.path.basename(i)
     os.makedirs("./data/val/dekopon/", exist_ok=True)
     shutil.move(i, f"./data/val/dekopon/{file_name}")

for i in grapefruit_data_train:
    file_name = os.path.basename(i)
    os.makedirs("./data/train/grapefruit/", exist_ok=True)
    shutil.move(i, f"./data/train/grapefruit/{file_name}")

for i in grapefruit_data_val:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/grapefruit/", exist_ok=True)
    shutil.move(i, f"./data/val/grapefruit/{file_name}")

for i in kanpei_data_train:
    file_name = os.path.basename(i)
    os.makedirs("./data/train/kanpei/", exist_ok=True)
    shutil.move(i, f"./data/train/kanpei/{file_name}")

for i in kanpei_data_val:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/kanpei/", exist_ok=True)
    shutil.move(i, f"./data/val/kanpei/{file_name}")

for i in orange_data_train:
    file_name = os.path.basename(i)
    os.makedirs("./data/train/orange/", exist_ok=True)
    shutil.move(i, f"./data/train/orange/{file_name}")

for i in orange_data_val:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/orange/", exist_ok=True)
    shutil.move(i, f"./data/val/orange/{file_name}")
