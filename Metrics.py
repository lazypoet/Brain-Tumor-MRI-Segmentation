# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:50:27 2016

@author: Shreyas_V
"""

import numpy as np

def accuracy(pred, orig_label, msk):
    '''  calculates complete accuracy
    INPUT: predicted labels, original labels, each 2D numpy arrays
    OUTPUT: float
    '''    
    acc = len(pred[(pred == orig_label) & msk])/float(len(msk[msk]))
    return acc

def DSC_en(pred, orig_label, cn, msk):
    TP = len(pred[(pred == cn) & (pred == orig_label) & msk])
    denom = len(pred[(pred == cn) & msk]) + len (orig_label[(orig_label == cn) & msk])
    if denom == 0:
        return -1
    return 2.*TP/float(denom)
    
def DSC(pred, orig_label):
    ''' Calculates Dice Score Coefficient
    INPUT: predicted, original labels
    OUTPUT: float
    '''
    TP = len(pred[((pred == 1) | (pred == 2) | (pred == 3) | (pred == 4)) & (pred == orig_label)])
    denom = len(pred[(pred == 1) | (pred == 2) | (pred == 3) | (pred == 4)]) + len(orig_label[(orig_label == 1) | (orig_label == 2) | (orig_label == 3) | (orig_label == 4)])
    if denom == 0:
        return -1
    return 2.*TP/float(denom)

def DSC_core_tumor(pred, orig_label):
    ''' Calculates DSC for core tumor (1, 3 and 4)
    INPUT: predicted, original labels
    OUTPUT: float
    '''
    TP = len(pred[((pred == 1) | (pred == 3) | (pred == 4)) & (pred == orig_label)])
    denom = len(pred[(pred == 1) | (pred == 3) | (pred == 4)]) + len(orig_label[(orig_label == 1) | (orig_label == 3) | (orig_label == 4)])
    if denom == 0:
        return -1
    return 2.*TP/float(denom)

def PPV(pred, orig_label):
    TP = len(pred[((pred == 1) | (pred == 3) | (pred == 4)) & (pred == orig_label)])
    FP = len(pred[((pred == 1) | (pred == 3) | (pred == 4)) & (pred != orig_label)])
    if TP == 0:
        return 0.
    return TP/float(TP+FP)