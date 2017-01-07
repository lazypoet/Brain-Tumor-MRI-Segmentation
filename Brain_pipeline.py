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
    ''' The Pipeline for loading images for all patients and all modalities, and preprocessing the images by:
        1) N4ITK bias correction   --DONE!
        2) Intensity normalization by Nyúl et al.  --DONE!
        
        INPUT: 
            1) The filepath 'path': Directory of the image database. It's subfolders contain the following .mha files:
            flair, t1, t1c, t2, and the ground truth(ground) MRI of all patients
            
    '''
    
    def __init__(self, path, n4 = 1, nyul = 1):
        self.path = path
        self.modes = ['flair', 't1', 't1c', 't2', 'ground']
        self.pathnames, self.gt_im, self.train_im = self.read_scans(n4, nyul)

    def read_scans(self,n4, nyul):
        if n4 == 1:
            flair = [flp for flp in glob(self.path + '*/*Flair*/*.mha') if 'n4' not in flp]
            t1 = [flp for flp in glob(self.path + '*/*T1[!c]*/*.mha') if 'n4' not in flp]
            t1c = [flp for flp in glob(self.path + '*/*T1c*/*.mha') if 'n4' not in flp]
            t2 = [flp for flp in glob(self.path + '*/*T2*/*.mha') if 'n4' not in flp]
            ground = [flp for flp in glob(self.path + '*/*more*/*.mha') if 'n4' not in flp]
            scans = [flair, t1, t1c, t2, ground]
            
            #scans has all the file pathnames stored. instead of loading all images in an array, let's do it one by one 
            #for N4 bias reduction atleast
            print('->Applying N4 Bias Correction...')
            for idx in xrange(4):
                for pat_nm in xrange(len(scans[idx])):
                    print 'patient directory {}; modality {}'.format(pat_nm, idx)
                    if os.path.exists( os.path.splitext(scans[idx][pat_nm])[0] + '_n4.mha'):
                        continue
                    tmp = sitk.ReadImage(scans[idx][pat_nm])
                    tmp = self.n4itk(tmp)       ##  Apply N4Bias Field Correction on all modalities
                    scans[idx][pat_nm] = os.path.splitext(scans[idx][pat_nm])[0] + '_n4.mha' #new pathname
                    sitk.WriteImage(tmp, scans[idx][pat_nm])
        else:
            flair = [flp for flp in glob(self.path + '*/*Flair*/*.mha') if 'n4.mha' in flp]
            t1 = [flp for flp in glob(self.path + '*/*T1[!c]*/*.mha') if 'n4.mha' in flp]
            t1c = [flp for flp in glob(self.path + '*/*T1c*/*.mha') if 'n4.mha' in flp]
            t2 = [flp for flp in glob(self.path + '*/*T2*/*.mha') if 'n4.mha' in flp]
            ground = [flp for flp in glob(self.path + '*/*more*/*.mha') if 'n4' not in flp]
            scans = [flair, t1, t1c, t2, ground]

        if nyul == 1:            
            print('->Applying Intensity Normalization...')
            for idx in xrange(4):
                im_arr = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in scans[idx]]    #load all images as arrays
                im_msk = [i>0 for i in im_arr]  #image masks
                normalizer = IntensityRangeStandardization()
                ret, out = normalizer.train_transform([i[m] for i, m in zip(im_arr, im_msk)])   #   Apply the Nyul Intensity Normaliztion on all modalities and standardize the image histograms
                for i, m, o, cnt in zip(im_arr, im_msk, out, xrange(len(scans[idx]))):
                    i[m] = o # redefine the 3D array of an image using output and mask
                    scans[idx][cnt] = scans[idx][cnt]+'_processed.mha'
                    print ('Processing applied: ' + cnt) 
                    sitk.WriteImage(sitk.GetImageFromArray(i), scans[idx][cnt])
        else:
            flair = [flp for flp in glob(self.path + '*/*Flair*/*.mha') if '.mha_processed' in flp]
            t1 = [flp for flp in glob(self.path + '*/*T1[!c]*/*.mha') if '.mha_processed' in flp]
            t1c = [flp for flp in glob(self.path + '*/*T1c*/*.mha') if '.mha_processed' in flp]
            t2 = [flp for flp in glob(self.path + '*/*T2*/*.mha') if '.mha_processed' in flp]
            ground = [flp for flp in glob(self.path + '*/*more*/*.mha') if 'n4' not in flp]
            scans = [flair, t1, t1c, t2, ground]
        
        #split pathnames for using in training - testing, in 67:33 ratio
        scans = np.array(scans)
        indices = np.random.permutation(scans.shape[1])
        train_idx = indices[:(scans.shape[1]*2)/3]
        test_idx = indices[(scans.shape[1]*2)/3:]
        self.pathnames_train = scans[:, train_idx]
        self.pathnames_test = scans[:, test_idx]
    
        gt_im = [sitk.GetArrayFromImage(sitk.ReadImage(i)) for i in self.pathnames_train[4]]
        train_im = []
        for i in xrange(4):
            train_im.append([sitk.GetArrayFromImage(sitk.ReadImage(i)) for i in self.pathnames_train[i]])
        return scans, np.array(gt_im), np.array(train_im)
    
    
    def n4itk(self, img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.Cast(img, sitk.sitkUInt8)   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
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
    def find_training_patches(self, num_patches, class_nm, d = 4, h = 33, w = 33):
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
        tmp_shp = self.gt_im.shape
        self.gt_im = self.gt_im.reshape(-1)
        # maintain list of 1D indices where label = class_nm
        indices = np.argwhere(self.gt_im == class_nm).reshape(-1)
        #reshape gt_im
        self.gt_im = self.gt_im.reshape(tmp_shp)
        st = timeit.default_timer()
        #find the patches from the images
        while count<num_patches:
            #print (count, ' cl:' ,class_nm)
            #sys.stdout.flush()
            #randomly choose an index
            ind = random.choice(indices)
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            #print ind
            #sys.stdout.flush()
            #patient directory to choose image from 
            pat_idx = ind[0]
            #find the slice index to choose from
            slice_idx = ind[1]
            #load the slice from the label
            l = self.gt_im[pat_idx][slice_idx]
           
            # the centre pixel and its coordinates
            p = ind[2:]
            #construct the patch by defining the coordinates
            p_x = (p[0] - h/2, p[0] + (h+1)/2)
            p_y = (p[1] - w/2, p[1] + (w+1)/2)
            #check if the pixels are in range
            if p_x[0]<0 or p_x[1]>l.shape[0] or p_y[0]<0 or p_y[1]>l.shape[1]:
                continue
            #take patches from all modalities and group them together
            tmp = []
            for idx in xrange(d):
                tmp.append(self.train_im[idx][pat_idx][slice_idx][p_x[0]:p_x[1], p_y[0]:p_y[1]])   #take the patch array from the image array
            tmp = np.array(tmp)
            patches.append(tmp)
            count+=1
        print 'finding patches of label {} took :'.format(class_nm), timeit.default_timer()-st
        return np.array(patches), labels
        

    def training_patches(self, num_patches, classes = 5, d = 4, h = 33, w = 33):
        '''Creates the input patches and their labels for training CNN. The patches are 4x33x33
    and the label for a patch corresponds to the label for the central pixel of the patch. The 
    data will be balanced, with the number of patches being the same for each class
            
            INPUT:
                    1) classes:  number of all classes in the segmentation
                    2) num_patches: number of patches for each class
                    3) d, h, w : channels, height and width of the patches
            OUTPUT:
                    1) all_patches: numpy array of all class patches of the shape 4x33x33
                    2) all_labels : numpy array of the all_patches labels
        '''
        
        patches, labels = [], []
        for idx in xrange(classes):
            p, l = self.find_training_patches(num_patches, idx, d, h, w)
            patches.append(p)
            labels.append(l)
        return np.array(patches).reshape(num_patches*classes, d, h, w), np.array(labels).reshape(num_patches*classes)
     
def test_patches(img ,d = 4, h = 33, w = 33):
    ''' Creates patches of image. Returns a numpy array of dimension number_of_patches x d.
    
            INPUT:
                    1)img: a 4D array containing the all modalities of a 2D image. 
                    2)d, h, w: see above
            OUTPUT:
                    tst_arr: ndarray of all patches of all modalities. Of the form number of patches x modalities
    '''
    
    #list of patches
    p = []
    for i in img:
        plist = extract_patches_2d(i, (h, w))
        p.append(np.array(plist))
    
    return zip(*np.array(p))

def reconstruct_labels(pred_list):
    pred = np.array(pred_list).reshape(208, 208)
    return np.pad(pred, (16, 16),  mode='edge')    