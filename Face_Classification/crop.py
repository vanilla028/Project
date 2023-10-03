import os
import glob
import cv2
from face_crop import face_crop

path = './dataset/'
crop_path = './crop/'

all_folder_path = glob.glob(os.path.join(path, '*', '*'))

for folder_path in all_folder_path:
    all_image_path = glob.glob(os.path.join(folder_path, '*.png'))
    for idx, image_path in enumerate(all_image_path):
        sex = folder_path.split('/')[2]
        age = folder_path.split('/')[3]
        save_path = crop_path + f'{sex}/{age}/'
        os.makedirs(save_path, exist_ok=True)

        img = face_crop(image_path)
        if img is not None:
            cv2.imwrite(save_path + f'{age}_{idx}.png', img)
