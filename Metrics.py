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

def DSC_en(pred, orig_label):
    TP = len(pred[(pred == 4) & (pred == orig_label)])
    denom = len(pred[(pred == 4) & msk]) + len (orig_label[(orig_label == 4)])
    if denom == 0:
        return 1.
    return 2.*TP/float(denom)
    
def DSC(pred, orig_label):
    ''' Calculates Dice Score Coefficient
    INPUT: predicted, original labels
    OUTPUT: float
    '''
    TP = len(pred[(pred != 0) & (orig_label != 0)])
    denom = len(pred[pred!=0]) + len(orig_label[orig_label != 0])
    if denom == 0:
        return 1
    return 2.*TP/float(denom)

def DSC_core_tumor(pred, orig_label):
    ''' Calculates DSC for core tumor (1, 3 and 4)
    INPUT: predicted, original labels
    OUTPUT: float
    '''
    TP = len(pred[((pred!=0) & (pred!=2)) & ((orig_label !=0) & (orig_label !=2))])
    denom = len(pred[pred[((pred!=0) & (pred!=2))]) + len(orig_label[(orig_label !=0) & (orig_label !=2)])
    if denom == 0:
        return 1
    return 2.*TP/float(denom)

def PPV(pred, orig_label):
    TP = len(pred[((pred!=0) & (pred!=2)) & ((orig_label !=0) & (orig_label !=2))])
    FP = len(pred[((pred!=0) & (pred!=2)) & ((orig_label ==0) | (orig_label ==2))])
    if TP == 0:
        return 0.
    return TP/float(TP+FP)
