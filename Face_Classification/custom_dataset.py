import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

data_path = os.path.join('./', 'dataset')

class MyCustomData(Dataset):
    def __init__(self, data_path, task, transforms=None):
        image_dir = os.path.join(data_path, task)
        image_path_list = glob.glob(os.path.join(image_dir, '*', '*.jpg'))

        self.image_path_list = image_path_list
        self.label_dict = {label: index for index, label in enumerate(os.listdir(image_dir))}
        self.transforms = transforms
        # print(self.label_dict)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)  # 버전 높아지니 필요 없음
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        temp_label = image_path.split('\\')[-2]
        label = self.label_dict[temp_label]
        return image, label

    def __len__(self):
        return len(self.image_path_list)

test = MyCustomData(data_path, 'train')
for i in test:
    pass
