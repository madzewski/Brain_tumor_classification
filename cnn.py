#Set-ExecutionPolicy Unrestricted -Scope Process
import joblib
from tqdm import tqdm

from utils.load_data import load_data, data_len
from utils.tools import split_data, plot_metrics, append_history
from utils.building_models import *


IMG_WIDTH = 240
IMG_HEIGHT = 240


def train_model(train_path, test_path, val_path, e = 10, b = 32, sample_size = 200):
    X_train, y_train = load_data(train_path, stop=sample_size, preprocess = False)
    X_val, y_val = load_data(val_path, stop=int(sample_size/10), preprocess = False)
    X_test, y_test = load_data(test_path, stop=int(sample_size/10), preprocess = False)

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of validation examples = " + str(X_val.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))

    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
    model = build_cnn_model_ver1(IMG_SHAPE)
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, batch_size=b, epochs=e, validation_data=(X_val, y_val))
    model.save('models/cnn_model_test.h5')
    history = model.history.history
    plot_metrics(history)
    model.evaluate(X_test, y_test)


def train_model_on_batch(train_path, test_path, val_path, e = 10, b = 32):
    # IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 1)
    best_acc = 0
    history = {'val_loss' : [], 'val_accuracy': []} 
    training_data_lenght = int(data_len(train_path)/2)

    model=build_cnn_model_ver2(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    for epoch in tqdm(range(e)):
        for batch in range(0,training_data_lenght,b):
            X_train, y_train = load_data(train_path, start = batch, stop=batch+b, preprocess = False, disable_tqdm = True)
            model.train_on_batch(X_train,y_train)
        if training_data_lenght%b != 0:
            X_train, y_train = load_data(train_path, start = training_data_lenght - (training_data_lenght%b), stop=training_data_lenght-1, preprocess = False, disable_tqdm = True)
            model.train_on_batch(X_train,y_train)
        
        X_val, y_val = load_data(val_path, start = 0, stop=200, preprocess = False, disable_tqdm = True)
        epoch_history = model.evaluate(X_val, y_val, return_dict=True)
        history = append_history(history,epoch_history)

        if float(epoch_history['accuracy']) > best_acc:
            print(f'Found better model. Acc: {epoch_history["accuracy"]}. Saving')
            model.save('models/cnn_model_best_v3.h5')
            best_acc = float(epoch_history['accuracy'])

    model.save('models/cnn_model_batch_long_v3.h5')
    X_test, y_test = load_data(test_path, preprocess = False)
    model.evaluate(X_test, y_test)
    plot_metrics(history)


train_path = './preprocessed_data/train'
test_path = './preprocessed_data/test'
val_path = './preprocessed_data/val'
# train_model(train_path = train_path, test_path = test_path, val_path = val_path, sample_size=800, e= 32)
train_model_on_batch(train_path = train_path, test_path = test_path, val_path = val_path, e= 42, b=8)