from tqdm import tqdm
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from .tools import preprocess_image


def load_data(dir_list, image_size = (240, 240), start = 0, stop = 9000, describe = False, preprocess = True, disable_tqdm = False):
    # load all images in a directory
    X = []
    y = []
    
    for path in tqdm(sorted(os.listdir(dir_list)), disable=disable_tqdm):
        for filename in tqdm(os.listdir(dir_list +'/'+ path)[start:stop], disable=disable_tqdm):
            image = cv2.imread(dir_list +'/'+ path +'/'+filename)
            if preprocess:
                X.append(preprocess_image(image, image_size=image_size))
            else:
                X.append(image/255.)
    
            if path[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    if describe:
        print(f'Number of examples is: {len(X)}')
        print(f'X shape is: {X.shape}')
        print(f'y shape is: {y.shape}')
    return X, y

def data_len(dir_list):
    lenght = 0
    for path in sorted(os.listdir(dir_list)):
        lenght += len(os.listdir(dir_list +'/'+ path))
    return lenght