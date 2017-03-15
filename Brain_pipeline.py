# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 01:47:13 2016

@author: seeker105
"""
import os.path
import random
import pylab
import numpy as np
from glob import glob
import SimpleITK as sitk
from Nyul import IntensityRangeStandardization
import sys
import timeit
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import color

class Pipeline(object):
    ''' The Pipeline for loading images for all patients and all modalities
        1)find_training_patches: finds the training patches for a particular class
        
        INPUT: 
            1) The filepath 'path': Directory of the image database. It contains the slices of training image slices
            
    '''
    
    def __init__(self, path_train = '', path_test = '' ):
        self.path_train = path_train
        self.path_test = path_test
        self.scans_train, self.scans_test, self.train_im, self.test_im = self.read_scans()
        
        
    def read_scans(self):
       
        scans_train = glob(self.path_train + r'/*.mha')
        scans_test = glob(self.path_test + r'/*.mha')
        train_im = [sitk.GetArrayFromImage(sitk.ReadImage(i)) for i in scans_train]
        test_im = [sitk.GetArrayFromImage(sitk.ReadImage(i)) for i in scans_test]
        return scans_train, scans_test, np.array(train_im), np.array(test_im)
    
    
    def n4itk(self, img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
        return corrected_img
    
    '''def find_all_train(self, classes, d = 4, h = 33, w = 33):
        mn = 300000000000000
        #load all labels
        im_ar = []
        for i in self.pathnames_train:
            im_ar.append([sitk.GetArrayFromImage(sitk.ReadImage(idx)) for idx in i])
        im_ar = np.array(im_ar)
        for i in xrange(classes):
            mn = min(mn, len(np.argwhere(im_ar[i]==i)))
    '''
    def sample_training_patches(self, num_patches, class_nm, d = 4, h = 33, w = 33):
        ''' Creates the input patches and their labels for training CNN. The patches are 4x33x33
        and the label for a patch equals to the label for the central pixel of the patch.
        
        INPUT:
            1) num_patches: The number of patches required of the class.
            
            2) class_nm:    The index of class label for which we are finding patches.
            
            3) d, h, w:     number of channels, height and width of patch
        
        OUTPUT:
            1) patches:     The list of all patches of dimensions d, h, w. 
            
            2) labels:      The list of labels for each patch. Label for a patch corresponds to the label
                            of the central pixel of that patch.
                            
        '''
        
        #find patches for training
        patches, labels = [], np.full(num_patches, fill_value = class_nm,  dtype = 'float')
        count = 0
        # convert gt_im to 1D and save shape
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   #swap axes to make axis 0 represent the modality and axis 1 represent the slice. take the ground truth
        #take flair image as mask
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        tmp_shp = gt_im.shape
        gt_im = gt_im.reshape(-1)
        msk = msk.reshape(-1)
        # maintain list of 1D indices where label = class_nm
        indices = np.squeeze(np.argwhere((gt_im == class_nm) & (msk != 0.)))
        # shuffle the list of indices of the class
        st = timeit.default_timer()
        np.random.shuffle(indices)
        print 'shuffling of label {} took :'.format(class_nm), timeit.default_timer()-st
        #reshape gt_im
        gt_im = gt_im.reshape(tmp_shp)
        st = timeit.default_timer()
        #find the patches from the images
        i = 0
        pix = len(indices)
        while (count<num_patches) and (pix>i):
            #print (count, ' cl:' ,class_nm)
            #sys.stdout.flush()
            #randomly choose an index
            ind = indices[i]
            i+= 1
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            #print ind
            #sys.stdout.flush()
            #find the slice index to choose from
            slice_idx = ind[0]
            #load the slice from the label
            l = gt_im[slice_idx]
           
            # the centre pixel and its coordinates
            p = ind[1:]
            #construct the patch by defining the coordinates
            p_x = (p[0] - h/2, p[0] + (h+1)/2)
            p_y = (p[1] - w/2, p[1] + (w+1)/2)
            #check if the pixels are in range
            if p_x[0]<0 or p_x[1]>l.shape[0] or p_y[0]<0 or p_y[1]>l.shape[1]:
                continue
            #take patches from all modalities and group them together
            tmp = self.train_im[slice_idx][0:4, p_x[0]:p_x[1], p_y[0]:p_y[1]]
            patches.append(tmp)
            count+=1
        print 'finding patches of label {} took :'.format(class_nm), timeit.default_timer()-st
        patches = np.array(patches)
        return patches, labels
        

    def training_patches(self, num_patches, classes = 5, d = 4, h = 33, w = 33):
        '''Creates the input patches and their labels for training CNN. The patches are 4x33x33
    and the label for a patch corresponds to the label for the central voxel of the patch. The 
    data will be balanced, with the number of patches being the same for each class
            
            INPUT:
                    1) classes:  number of all classes in the segmentation
                    2) num_patches: number of patches for each class
                    3) d, h, w : channels, height and width of the patches
            OUTPUT:
                    1) all_patches: numpy array of all class patches of the shape 4x33x33
                    2) all_labels : numpy array of the all_patches labels
        '''
        
        patches, labels, mu, sigma = [], [], [], []
        for idx in xrange(classes):
            p, l = self.sample_training_patches(num_patches, idx, d, h, w)
            patches.append(p)
            labels.append(l)
        patches = np.array(patches).reshape(num_patches*classes, d, h, w) 
        patches_by_channel = np.swapaxes(patches, 0, 1)
        for seq, i in zip(patches_by_channel, xrange(d)):
            avg = np.mean(seq, dtype = np.float64)
            std = np.std(seq, dtype = np.float64)
            patches_by_channel[i] = (patches_by_channel[i] - avg)/std
            mu.append(avg)
            sigma.append(std)
        patches = np.swapaxes(patches_by_channel, 0, 1)
        return patches, np.array(labels).reshape(num_patches*classes), np.array(mu), np.array(sigma)
     
def test_patches(img , mu = 0, sigma = 1, d = 4, h = 33, w = 33):
    ''' Creates patches of image. Returns a numpy array of dimension number_of_patches x d.
    
            INPUT:
                    1)img: a 3D array containing the all modalities of a 2D image. 
                    2)d, h, w: see above
            OUTPUT:
                    tst_arr: ndarray of all patches of all modalities. Of the form number of patches x modalities
    '''
    
    #list of patches
    p = []
    msk = (img[0]!=0)   #mask using FLAIR channel
    msk = msk[h/2:-(h/2), w/2:(-w/2)]      #crop the mask to conform to the rebuilt image after prediction
    msk = msk.reshape(-1)
    for i in xrange(len(img)):
        img[i] = (img[i] - mu[i])/sigma[i]
        plist = extract_patches_2d(img[i], (h, w))
        if len(plist[np.where(msk)]) == 0:
            return -1
        p.append(plist[np.where(msk)])              #only take patches with brain mask!=0
    return np.array(p).swapaxes(0, 1)
    

def reconstruct_labels(im, msk, pred_list):
    im[msk] = np.array(pred_list)
    return im    