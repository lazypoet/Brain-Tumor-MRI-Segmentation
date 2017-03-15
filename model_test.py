# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 17:08:10 2017

@author: Shreyas_V
"""
import Brain_pipeline
import Metrics
import numpy as np

#For testing a model

def predict_labels(test_images, model, mu, sigma):
    '''
    INPUT: a numpy array of 4x240x240 elements and a keras model already trained over various images
    OUTPUT: a numpy array of 240x240 label elements
    '''
    predicted_images = []   # list to maintain predicted labels
    for i in test_images:
        patches = Brain_pipeline.test_patches(i, mu, sigma)
        if patches == -1:
            predicted_images.append(np.full((240, 240), 0))
            continue
        print "running..."
        predicted_slice = model.predict_classes(patches)
        predicted_slice = Brain_pipeline.reconstruct_labels(predicted_slice)
        predicted_images.append(predicted_slice)
    return np.array(predicted_images)
    
def get_metrics(test_images, gt):
    DSC = []
    acc = []
    DSC_core = []
    PPV = []
    for i, j in zip(test_images, gt):
        DSC.append(Metrics.DSC(i, j))
        acc.append(Metrics.accuracy(i, j))
        DSC_core.append(Metrics.DSC_core_tumor(i, j))
        PPV.append(Metrics.PPV(i, j))
    return DSC, acc, DSC_core, PPV

def test_slices(test_images, gt, model, mu=0, sigma=1):
    pred = predict_labels(test_images, model, mu, sigma)
    return pred, get_metrics(pred, gt)

    