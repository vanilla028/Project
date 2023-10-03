import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def data_split(data_path):
    image_format_list = ['jpeg', 'jpg', 'png'] 
    
    image_label_list = os.listdir(data_path)
    
    for label_name in image_label_list:
        image_path_list = glob.glob(os.path.join(data_path, label_name, '*'))
        
        train_data, temp_data = train_test_split(image_path_list, test_size=0.2, shuffle=True, random_state=20230110)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=20230110)
        
        temp_dict = {
            'train': train_data,
            'valid': valid_data,
            'test': test_data
        }
        
        for task, data in temp_dict.items():
            copy_data(task, data)
            

def copy_data(task, data):
    for index, image_path in enumerate(data):
        label = image_path.split('\\')[-2]
        
        dataset_dir_path = os.path.join('../', 'dataset', task, label)
        os.makedirs(dataset_dir_path, exist_ok=True)
        
        image_full_path = os.path.join(dataset_dir_path, '{}_{}.png'.format(label, index))
        image = make_square_image(image_path)

        if image is None:
            continue

        cv2.imwrite(image_full_path, image)
        
        
def make_square_image(image_path):
    origin_image = cv2.imread(image_path)
    
    try:
        height, width, channels = origin_image.shape
    except:
        print(image_path, '#' * 10)
        return
    
    x = height if height > width else width
    y = height if height > width else width
    
    if 224 > x and 224 > y:
        x, y = 224, 224
    
    square_image = np.zeros((x, y, channels), np.uint8) # 검정색 이미지 생성
    square_image[int((y - height) / 2):int(y - (y - height) / 2), 
                int((x - width) / 2):int(x - (x - width) / 2)] = origin_image
    
    return squre_image


data_path = os.path.join('./', 'data')
data_split(data_path)
