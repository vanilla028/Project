import glob
import os
from torch.utils.data import Dataset
import cv2

label_dic = {"cat": 0, "dog": 1}

class MyCustomDataset(Dataset):

    def __init__(self, data_path, transforms=None):
        self.all_data_path = glob.glob(os.path.join(data_path, '*', '*.jpg'))
        self.transforms = transforms
    

    def __getitem__(self, index):

        # 이미지 경로
        image_path = self.all_data_path[index]

        # feature 추출
        image = cv2.imread(image_path)

        # label 추출
        # "./archive/train\cats\cat_100.jpg'

        label_temp = image_path.split("\\")
        # './archive/train', 'cats', 'cat_100.jpg'
        
        label_temp = label_temp[2]
        label_temp = label_temp.split("_")
        # 'cat', '100.jpg'
        
        label_temp = label_temp[0]
        # 'cat' ---> 레이블 추출 완료!

        # 추출된 레이블을 숫자로 매핑
        label = label_dic[label_temp]
     
        # augmentation  적용 후
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.all_data_path)


# 레이블 추출 테스트
test = MyCustomDataset("./archive/train")
for i in range(10):
    _, label = test[i]
    print(label)
