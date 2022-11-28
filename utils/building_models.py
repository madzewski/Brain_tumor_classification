from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D, Dropout
from utils.tools import RGB2GrayTransformer, HogTransformer, ShapeTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_cnn_model_ver1(input_shape):
    X_input = Input(input_shape) 
    X = ZeroPadding2D((2, 2))(X_input) 
    
    X = Conv2D(32, (7, 7), strides = (1, 1))(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    
    X = MaxPooling2D((4, 4))(X) 
    X = MaxPooling2D((4, 4))(X) 
    X = Flatten()(X) 
    X = Dense(1, activation='sigmoid')(X) 
    X = Dropout(0.5)(X)
    model = Model(inputs = X_input, outputs = X)
    return model


def build_cnn_model_ver2(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X_input)
    X = Dropout(0.25)(X)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = Dropout(0.25)(X)
    X = MaxPooling2D()(X)

    X = Conv2D(32, (5,5), 1, activation="relu", padding="same")(X)
    X = Dropout(0.25)(X)
    X = Conv2D(32, (5,5), 1, activation="relu", padding="same")(X)
    X = Dropout(0.25)(X)
    X = MaxPooling2D()(X)

    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = Dropout(0.25)(X)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = Dropout(0.25)(X)
    X = MaxPooling2D()(X)

    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.35)(X)
    X = Dense(1, activation='sigmoid')(X)
    X = Dropout(0.35)(X)
    model = Model(inputs = X_input, outputs = X)
    return model


def build_svm_model(shape):
    model = Pipeline([
        # ('grayify', RGB2GrayTransformer()),
        ('reshape', ShapeTransformer(shape = shape)),
        ('scalify', StandardScaler()), 
        ('classify', svm.SVC(kernel='linear', C=1))
    ])
    return model


def build_pca_svm_model(shape):
    model = Pipeline([
        # ('grayify', RGB2GrayTransformer()),
        ('reshape', ShapeTransformer(shape = shape)),
        ('scaling', StandardScaler()),
        ('reduce_dim', PCA(n_components=0.9)),
        ('classify', svm.SVC(kernel='linear', C=1))
    ])
    return model


def build_hog_svm_model(shape):
    model = Pipeline([
        # ('grayify', RGB2GrayTransformer()),
        ('reshape', ShapeTransformer(shape = shape)),
        ('hogify', HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2, 2), 
            orientations=8, 
            block_norm='L2-Hys')
        ),
        ('scalify', StandardScaler()),
        ('classify', svm.SVC(kernel='linear', C=1,probability = True))
    ])
    return model