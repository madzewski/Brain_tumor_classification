import cv2
import imutils
import os
import numpy as np
import skimage
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    #Simple rgb to grayscale transformation
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    #Expects an array of 2d arrays (1 channel images)
    #Calculates hog features for each img
    def __init__(self, y=None, orientations=9,pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


def crop_brain_contour(image, grayscale = True):
    # Convert the image to grayscale, and blur it slightly
    if grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        gray = cv2.GaussianBlur(image, (5, 5), 0)
    
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]               
    return new_image


def preprocess_image(image, greyscale = True, image_size = (240, 240)):
    image_width, image_height = image_size
    image = crop_brain_contour(image, greyscale)
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    return image / 255.


def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test


def rename_images(path, name, start = 0):
    files = os.listdir(path)
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([name+str(index+start), '.jpg'])))


def append_history(history, temp_history):
    history['val_loss'].append(temp_history['loss'])
    history['val_accuracy'].append(temp_history['accuracy'])
    return history


def plot_metrics(history):
    # train_loss = history['loss']
    val_loss = history['val_loss']
    # train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    # Loss
    plt.figure()
    # plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    # Accuracy
    plt.figure()
    # plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def save_model_params(parameters):
    with open('file.txt', 'w') as file:
        file.write(json.dumps(parameters))
