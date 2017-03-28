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
from keras.initializers import constant
from keras.regularizers import l1_l2
from keras.optimizers import SGD
from keras import backend as K


class LeNet:
    @staticmethod
    def build_Pereira(w, h, d, classes, weightsPath = None, alp = 0.333, dropout = 0.1):
        '''INPUT:
                INPUT WIDTH, HEIGHT, DEPTH, NUMBER OF OUTPUT CLASSES, PRELOADED WEIGHTS, PARAMETER FOR LEAKYReLU, DROPOUT PROBABILITY
           OUTPUT:
                TRAINED CNN ARCHITECTURE
                '''
        K.set_image_dim_ordering('th')
        model = Sequential()
        
                
        #first set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (d, h, w), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #second set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha = alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #Fully connected layers
        
        # FC => LReLU => FC => LReLU
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        model.add(Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        
        # FC => SOFTMAX
        model.add(Dense(classes, kernel_initializer = 'glorot_normal', bias_initializer = constant(0.1)))
        model.add(Activation("softmax"))
        
        #if a pre-trained model is applied, load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model
    
    @staticmethod    
    def build_Nikki(w, h, d, classes, l1=0.01, l2=0.01):
        
        K.set_image_dim_ordering('th')
        model = Sequential()
        model.add(Convolution2D(64, (7, 7), activation = 'relu', kernel_regularizer=l1_l2(), input_shape = (d, h, w) ))
        model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        model.add(Dropout(0.5))
        model.add(Convolution2D(128, (5, 5), activation = 'relu', kernel_regularizer=l1_l2() ))
        model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        model.add(Dropout(0.5))
        model.add(Convolution2D(128, (5, 5), activation = 'relu', kernel_regularizer=l1_l2() ))
        model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, (3, 3), activation = 'relu', kernel_regularizer=l1_l2() ))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(5))
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model
        
        