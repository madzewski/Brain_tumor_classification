from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D


def build_cnn_model_ver1(input_shape):
    X_input = Input(input_shape) 
    X = ZeroPadding2D((2, 2))(X_input) 
    
    X = Conv2D(32, (7, 7), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    
    X = MaxPooling2D((4, 4))(X) 
    X = MaxPooling2D((4, 4))(X) 
    X = Flatten()(X) 
    X = Dense(1, activation='sigmoid')(X) 
    model = Model(inputs = X_input, outputs = X)
    
    return model


def build_cnn_model_ver2(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X_input)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = MaxPooling2D()(X)

    X = Conv2D(32, (5,5), 1, activation="relu", padding="same")(X)
    X = Conv2D(32, (5,5), 1, activation="relu", padding="same")(X)
    X = MaxPooling2D()(X)

    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = Conv2D(16, (3,3), 1, activation="relu", padding="same")(X)
    X = MaxPooling2D()(X)

    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = X)
    return model
