# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 11:35:51 2017

@author: Shreyas_V
"""

'''
Visualizes the weights of a keras model's layers, keeping the dimensions intact'''

import theano
import numpy as np
import numpy.ma as ma
import pylab

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def visualize_layer_output(model, layer_num, inp, nr=6, nc=6):
    '''
    Given a keras model, a layer number, a particular input for the model,
    shows a mosaic image of all filters in that layer with the image dimension of nr x nc,
    activated by the model's input given to the first layer
    '''
    out = theano.function([model.get_input(train=False)], model.layers[layer_num].get_output(train=False))
    op = out(inp)
    op = np.squeeze(op)
    pylab.imshow(make_mosaic(op, nr, nc), 'gray')
    
def visualize_layer(model, layer_num, nr=6, nc=6):
    '''
    Given a model and its layer number,
    shows a mosaic image of all the filter weights, 
    with the image dimension being nr x nc
    '''
    W = model.layers[layer_num].W.get_value(borrow=True)
    W = np.squeeze(W)
    pylab.imshow(make_mosaic(W, nr, nc))