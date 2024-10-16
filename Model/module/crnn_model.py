# import our model, different layers and activation function
import os
import tensorflow as tf
from tensorflow import keras
# from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
# from keras.models import Model
# import keras.backend as K


# Mô hình CRNN và LSTM nhận dạng ký tự

# input
inputs = keras.layers.Input(shape=(118, 2167, 1))
    
    # Block 1
x = keras.layers.Conv2D(64, (3,3), padding='same')(inputs)
x = keras.layers.MaxPool2D(pool_size=3, strides=3)(x)
x = keras.layers.Activation('relu')(x)
x_1 = x 

# Block 2
x = keras.layers.Conv2D(128, (3,3), padding='same')(x)
x = keras.layers.MaxPool2D(pool_size=3, strides=3)(x)
x = keras.layers.Activation('relu')(x)
x_2 = x

# Block 3
x = keras.layers.Conv2D(256, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x_3 = x

    # Block 4
x = keras.layers.Conv2D(256, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Add()([x,x_3])
x = keras.layers.Activation('relu')(x)
x_4 = x

# Block 5
x = keras.layers.Conv2D(512, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x_5 = x

# Block 6
x = keras.layers.Conv2D(512, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Add()([x,x_5])
x = keras.layers.Activation('relu')(x)

    # Block 7
x = keras.layers.Conv2D(1024, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 1))(x)
x = keras.layers.Activation('relu')(x)

    # Pooling layer
x = keras.layers.MaxPool2D(pool_size=(3, 1))(x)
    
    # Remove first dimension
squeezed = keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(x)
    
    # Bidirectional LSTM layers
blstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

    # Output layer
outputs = keras.layers.Dense(141, activation = 'softmax')(blstm_2)

# model to be used at test time
model = keras.models.Model(inputs, outputs)

# read model
model.load_weights('./data/model_checkpoint_weights.hdf5')


