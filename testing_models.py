import joblib
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D
from sklearn.metrics import accuracy_score
from keras.models import load_model
from utils.load_data import load_data
from utils.tools import split_data, plot_metrics



IMG_WIDTH = 240
IMG_HEIGHT = 240

test_path = './preprocessed_data/test'
X_test, y_test = load_data(test_path, preprocess = False)

model = load_model('models/cnn_model_best_v3.h5')
model.evaluate(X_test, y_test)



