#Set-ExecutionPolicy Unrestricted -Scope Process
import joblib
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from skimage.feature import hog
import pandas as pd

from utils.load_data import load_data, data_len
from utils.tools import split_data, plot_metrics, append_history, hog_transform
from utils.building_models import *


IMG_WIDTH = 240
IMG_HEIGHT = 240


def train_model_on_batch_with_hog(train_path, test_path, val_path, e = 6, b = 8):
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT,1)
    best_acc = 0
    history = {'val_loss' : [], 'val_accuracy': []} 
    training_data_lenght = int(data_len(train_path)/2)

    model=build_cnn_model_ver1(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    for epoch in tqdm(range(e)):
        for batch in range(0,training_data_lenght,b):
            X_train, y_train = load_data(train_path, start = batch, stop=batch+b, preprocess = False, disable_tqdm = True)
            X_train = hog_transform(X_train)
            model.train_on_batch(X_train,y_train)
        if training_data_lenght%b != 0:
            X_train, y_train = load_data(train_path, start = training_data_lenght - (training_data_lenght%b), stop=training_data_lenght-1, preprocess = False, disable_tqdm = True)
            X_train = hog_transform(X_train)
            model.train_on_batch(X_train,y_train)
        
        X_val, y_val = load_data(val_path, preprocess = False, disable_tqdm = True)
        X_val = hog_transform(X_val)
        epoch_history = model.evaluate(X_val, y_val, return_dict=True)
        history = append_history(history,epoch_history)

        if float(epoch_history['accuracy']) > best_acc:
            print(f'Found better model. Acc: {epoch_history["accuracy"]}. Saving')
            model.save('models/cnn_model_v1_best_short_hog.h5')
            best_acc = float(epoch_history['accuracy'])

    model.save('models/cnn_model_v1_short_hog.h5')
    X_test, y_test = load_data(test_path, preprocess = False)
    X_test = hog_transform(X_test)
    model.evaluate(X_test, y_test)
    plot_metrics(history)


def train_model_on_batch(train_path, test_path, val_path, e = 10, b = 32):
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT,1)
    best_acc = 0
    history = {'val_loss' : [], 'val_accuracy': []} 
    training_data_lenght = int(data_len(train_path)/2)

    model=build_cnn_model_ver3(IMG_SHAPE)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    
    for epoch in tqdm(range(e)):
        for batch in range(0,training_data_lenght,b):
            X_train, y_train = load_data(train_path, start = batch, stop=batch+b, preprocess = False, disable_tqdm = True)
            model.train_on_batch(X_train,y_train)
        if training_data_lenght%b != 0:
            X_train, y_train = load_data(train_path, start = training_data_lenght - (training_data_lenght%b), stop=training_data_lenght-1, preprocess = False, disable_tqdm = True)
            model.train_on_batch(X_train,y_train)
        
        X_val, y_val = load_data(val_path, preprocess = False, disable_tqdm = True)
        epoch_history = model.evaluate(X_val, y_val, return_dict=True)
        history = append_history(history,epoch_history)

        if float(epoch_history['accuracy']) > best_acc:
            print(f'Found better model. Acc: {epoch_history["accuracy"]}. Saving')
            model.save('models/vgg_best_dropout_aug.h5')
            best_acc = float(epoch_history['accuracy'])

    model.save('models/vgg_.h5')
    X_test, y_test = load_data(test_path, preprocess = False)
    model.evaluate(X_test, y_test)
    df = pd.DataFrame.from_dict(history)
    df.to_csv('vgg.csv',sep =',', index = False)
    plot_metrics(history)


train_path = './augmented_preprocessed_data/train'
test_path = './augmented_preprocessed_data/test'
val_path = './augmented_preprocessed_data/val'
# train_model_on_batch_with_hog(train_path = train_path, test_path = test_path, val_path = val_path, e= 64, b=16)
train_model_on_batch(train_path = train_path, test_path = test_path, val_path = val_path, e= 20, b=6)