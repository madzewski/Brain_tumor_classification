from tqdm import tqdm
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from tools import crop_brain_contour


def preprocess_images(dir_list, save_path, image_size = (240, 240), start = 0, stop = 100):
    image_width, image_height = image_size
    
    for path in tqdm(sorted(os.listdir(dir_list))):
        for filename in tqdm(os.listdir(dir_list +'/'+ path)[start:stop]):
            image = cv2.imread(dir_list +'/'+ path +'/'+filename)
            image = crop_brain_contour(image)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(save_path + '/' + path + '/'+filename, image)

def augment_images(path, save_path, start = 0, stop = 100):
    counter = 1250
    for file in tqdm(sorted(os.listdir(path))[start:stop]):
        image = cv2.imread(path +'/'+file)
        image = cv2.flip(image, 1)
        cv2.imwrite(save_path + '/n'+str(counter)+'.jpg', image)
        counter += 1
# preprocess_images(dir_list='../data', save_path='../preprocessed_data', stop = 5000)
augment_images(path='../preprocessed_data/train/no', save_path= '../preprocessed_data/train/no', stop=800)