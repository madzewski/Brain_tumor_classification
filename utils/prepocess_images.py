from tqdm import tqdm
import time
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
            # image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            print('xD')
            cv2.imwrite(save_path + '/' + path + '/'+filename, image)
            return

def rename_files(path, name):
    for counter, filename in enumerate(tqdm(sorted(os.listdir(path)))):
        dst = f"{name}{str(counter)}.jpg"
        src =f"{path}/{filename}"
        dst =f"{path}/{dst}"
        os.rename(src, dst)


def augment_images(path, save_path, start = 0, stop = 100):
    counter = 0
    for file in tqdm(sorted(os.listdir(path))[start:stop]):
        image = cv2.imread(path +'/'+file)
        cv2.imwrite(save_path + '/' + file[0:1]+str(counter)+'.jpg', image)
        counter += 1
    time.sleep(1)
    for file in tqdm(sorted(os.listdir(path))[start:stop]):
        image = cv2.imread(path +'/'+file)
        image = cv2.flip(image, 1)
        cv2.imwrite(save_path + '/' + file[0:1]+str(counter)+'.jpg', image)
        counter += 1
    time.sleep(1)
    for file in tqdm(sorted(os.listdir(path))[start:stop]):
        image = cv2.imread(path +'/'+file)
        image = cv2.flip(image, 0)
        cv2.imwrite(save_path + '/' + file[0:1] +str(counter)+'.jpg', image)
        counter += 1
    time.sleep(1)
    for file in tqdm(sorted(os.listdir(path))[start:stop]):
        image = cv2.imread(path +'/'+file)
        image = cv2.flip(image, 1)
        image = cv2.flip(image, 0)
        cv2.imwrite(save_path + '/' + file[0:1] +str(counter)+'.jpg', image)
        counter += 1
    
if __name__ == "__main__":
    path = '../augmented_preprocessed_data/train/yes'
    save_path = 'D:\praca_inz'
    # preprocess_images(dir_list='../data', save_path='D:\praca_inz', stop = 5000)
    augment_images(path=path, save_path=save_path,start=0, stop=1)
    # rename_files(path = path, name = 'n')