# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 11:36:23 2016

@author: seeker105
"""

''' First CNN training and testing '''

from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import backend as K


class LeNet:
    @staticmethod
    def build(w, h, d, classes, weightsPath = None, alp = 0.3, dropout = 0.1):
        '''INPUT:
                INPUT WIDTH, HEIGHT, DEPTH, NUMBER OF OUTPUT CLASSES, PRELOADED WEIGHTS, PARAMETER FOR LEAKYReLU, DROPOUT PROBABILITY
           OUTPUT:
                TRAINED CNN ARCHITECTURE
                '''
        K.set_image_dim_ordering('th')
        model = Sequential()
        
        # normalize the patches of each modality to have zero mean and unit variance
        model.add(BatchNormalization(mode=0, axis=1, input_shape=(d, h, w)))
        
        #first set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(64, 3, 3, border_mode="same", input_shape = (d, h, w)))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, 3, 3,border_mode="same", input_shape = (33, 33, 64)))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, 3, 3, border_mode="same",input_shape = (33, 33, 64)))
        model.add(LeakyReLU(alpha=alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #second set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(128, 3, 3, border_mode="same",input_shape = (16, 16, 64)))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, 3, 3, border_mode="same",input_shape = (16, 16, 128)))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, 3, 3, border_mode="same",input_shape = (16, 16, 128)))
        model.add(LeakyReLU(alpha = alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #Fully connected layers
        
        # FC => LReLU => FC => LReLU
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(alp))
        model.add(Dropout(dropout))
        model.add(Dense(256))
        model.add(LeakyReLU(alp))
        model.add(Dropout(dropout))
        
        # FC => SOFTMAX
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        #if a pre-trained model is applied, load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model
        
        