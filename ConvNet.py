# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 11:36:23 2016

@author: seeker105
"""

''' First CNN training and testing '''

from keras.models import Sequential
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, merge, Convolution2D, MaxPooling2D
from keras.layers import UpSampling2D, Activation, Dropout
from keras.layers import Dense, Flatten, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.initializers import constant
from keras.regularizers import l1_l2
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nnet.abstract_conv import bilinear_upsampling

#bilinear upsampling layer
class Deconv2D(Layer):
    def __init__(self, ratio, **kwargs):
        self.ratio = ratio
        super(Deconv2D, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Deconv2D,self).build(input_shape)
    def call(self, x, mask=None):
        return bilinear_upsampling(x, ratio=self.ratio)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],
                input_shape[2] * self.ratio, input_shape[3] * self.ratio)

        
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
        
    @staticmethod
    def unet(w, h, d):
        K.set_image_dim_ordering('th')

        inputs = Input((d, h, w))
        conv1 = Convolution2D(32, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(inputs)
        conv1 = Convolution2D(32, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Convolution2D(64, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(pool1)
        conv2 = Convolution2D(64, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same', data_format = 'channels_first')(pool2)
        conv3 = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Convolution2D(256, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(pool3)
        conv4 = Convolution2D(256, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same', data_format = 'channels_first')(pool4)
        conv5 = Convolution2D(512, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv5)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
        # convdeep = Convolution2D(1024, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(pool5)
        # convdeep = Convolution2D(1024, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(convdeep)
        
        # upmid = merge([Convolution2D(512, kernel_size = (2, 2), padding='same', data_format = 'channels_first')(UpSampling2D(size=(2, 2), data_format = 'channels_first')(convdeep)), conv5], mode='concat', concat_axis=1)
        # convmid = Convolution2D(512, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(upmid)
        # convmid = Convolution2D(512, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(convmid)
    
        up6 = merge([Convolution2D(256, kernel_size = (2, 2),activation='relu', padding='same', data_format = 'channels_first')(UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv5)), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(256, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(up6)
        conv6 = Convolution2D(256, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv6)
    
        up7 = merge([Convolution2D(128, kernel_size = (2, 2),activation='relu', padding='same', data_format = 'channels_first')(UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv6)), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(128, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(up7)
        conv7 = Convolution2D(128, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv7)
    
        up8 = merge([Convolution2D(64, kernel_size = (2, 2),activation='relu', padding='same', data_format = 'channels_first')(UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv7)), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(64, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(up8)
        conv8 = Convolution2D(64, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv8)
    
        up9 = merge([Convolution2D(32, kernel_size = (2, 2),activation='relu', padding='same', data_format = 'channels_first')(UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv8)), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(32, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(up9)
        conv9 = Convolution2D(32, kernel_size = (3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv9)
    
        conv10 = Convolution2D(5, kernel_size=(1, 1), padding='same', data_format = 'channels_first')(conv9)
        flat = Reshape((5, h*w))(conv10)
        out = Activation('softmax')(flat)
        model = Model(input=inputs, output=out)
        model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
        return model

    
    #let's see how a Conv Net without any pooling layers fares, as pooling layers are known to reduce data, but stop overfitting
    @staticmethod
    def build_Pereira_mod(w, h, d, classes, weightsPath = None, alp = 0.333, dropout = 0.1):
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
        
        #second set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha = alp))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        
        #Fully connected layers
        
        # FC => LReLU => FC => LReLU
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(128, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        model.add(Dense(128, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
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
    def build_Pereira_no_pooling(w, h, d, classes, weightsPath = None, alp = 0.333, dropout = 0.1):
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
        model.add(Convolution2D(128, kernel_size=(3, 3), strides = (2, 2), padding="valid", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))

        #second set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha = alp))
        model.add(Convolution2D(256, kernel_size=(3, 3), strides = (2, 2), padding="valid", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        #Fully connected layers
        
        # FC => LReLU => FC => LReLU
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(128, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        model.add(Dense(128, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        
        # FC => SOFTMAX
        model.add(Dense(classes, kernel_initializer = 'glorot_normal', bias_initializer = constant(0.1)))
        model.add(Activation("softmax"))
        
        #if a pre-trained model is applied, load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model