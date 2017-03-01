# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 21:35:59 2016

@author: seeker105
"""
import os.path
import sys
from ConvNet import LeNet
import json
import SimpleITK as sitk
import pylab
from skimage import color
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
from Brain_pipeline import Pipeline
import Brain_pipeline
import Metrics
from glob import glob
import model_test

''' Script to drive loading, training, testing and saving the brain MRI
    First, we load all the images, and process them through the Pipeline,
    and get the pre-processed images as output.
    Then, we train the model from ConvNet or load the weights into it.
    We divide the training and test data using train_test_split.
    '''
def show_segmented_image(orig_img, pred_img):
    '''
    Show the prediction over the original image
    INPUT:
        1)orig_img: the test image, which was used as input
        2)pred_img: the prediction output
    OUTPUT:
        segmented image rendering
    '''
    #define the colours of the labels
    red = [10, 0, 0] #label 1
    yellow = [10, 10, 0] #label 2
    green = [0, 10, 0]  #label 3
    blue = [0, 0, 10] #label 4
    #convert original image to rgb
    gray_im = color.gray2rgb(orig_img)
    #color the tumor voxels
    gray_im[pred_img == 1] = red
    gray_im[pred_img == 2] = yellow
    gray_im[pred_img == 3] = green
    gray_im[pred_img == 4] = blue
    pylab.imshow(gray_im)


def step_decay(epochs):
    init_rate = 0.003
    fin_rate = 0.00003
    total_epochs = 24
    print 'ep: {}'.format(epochs)
    if epochs<25:
        lrate = init_rate - (init_rate - fin_rate)/total_epochs * float(epochs)
    else: lrate = 0.00003
    print 'lrate: {}'.format(lrate)
    return lrate

pth_train = 'D:/New folder/BRATS2015_Training/train_slices/'
pth_test = 'D:/New folder/BRATS2015_Training/test_slices/'
x = Pipeline(pth_train, pth_test)   #pass the images through the preprocessing steps

#build the model
model = LeNet.build_Pereira(33, 33, 4, 5)   

#callback
change_lr = LearningRateScheduler(step_decay)

#initialize the optimizer and model
opt = SGD(lr = 0.003, momentum=0.9, decay= 0, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])


#load training patches
X_patches, Y_labels = x.training_patches(120000)    
# Labels should be in categorical array form 1x5
Y_labels = np_utils.to_categorical(Y_labels, 5)


#save model after each epoch
os.mkdir(r'D:\New folder\Pereira_model_checkpoints')
checkpointer = ModelCheckpoint(filepath='D:/New folder/Pereira_model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor = 'val_loss', verbose=1)
#fit model and shuffle training data
hist = model.fit(X_patches, Y_labels, nb_epoch=25, batch_size=128, verbose=1, shuffle=True, validation_split=0.01, callbacks = [change_lr, checkpointer])
 
#save model
sv_pth = 'D:/New Folder/Pereira_model_checkpoints/model_weights'
m = '{}.json'.format(sv_pth)
w = '{}.hdf5'.format(sv_pth)
model.save_weights(w)
json_strng = model.to_json()
with open(m, 'w') as f:
    json.dump(json_strng, f)
    

#test all the test image slices
test_im = x.test_im.swapaxes(0,1)
gt = test_im[:4]
test_im = test_im[4]
predicted_images, DSC, acc, DSC_core, PPV = model_test.test_slices(test_im, gt, model)




'''test_pths = zip(*x.pathnames_test)
#show a segmented slice
tst = test_pths[0]#random.choice(test_pths)
test_arr = [sitk.GetArrayFromImage(sitk.ReadImage(i)) for i in tst]
final_pth = os.path.dirname(os.path.dirname(tst[0])) +  '/' + os.path.splitext(os.path.splitext(os.path.basename(tst[0]))[0])[0] + '_processed_predicted_70.mha'  
slice_arr = [test_arr[j][70] for j in xrange(4)]
patches = Brain_pipeline.test_patches(slice_arr)
pred = model.predict_classes(patches)
pred = Brain_pipeline.reconstruct_labels(pred)
show_segmented_image(test_arr[0][70], pred)
sitk.WriteImage(sitk.GetImageFromArray(np.array(pred.astype(float))), final_pth)


#evaluate metrics
DSC_arr = [] #stores DSC
DSC_core_arr = [] #stores list of core DSCs
PPV_arr = []
acc_arr = []

#use for getting orignal brain image and prediction label slices
# use for:
    #overlay images
    #segmentation vs orig label. it's in test_paths
    #with/without nyul
    #4 sequences after nyul. for original ones, redefine paths
    #ok. now we gotta see metrics brother
pred_pth = []
t1c_pth = []
pred_arr = []

for i in xrange(len(test_pths)):
    tst = test_pths[i]
    test_arr = [sitk.GetArrayFromImage(sitk.ReadImage(j)) for j in tst]
    #take slices
    slice_arr = [test_arr[j][70] for j in xrange(4)]
    #read original slice label
    orig = test_arr[4][70]
    patches = Brain_pipeline.test_patches(slice_arr)
    pred = model.predict_classes(patches)
    pred = Brain_pipeline.reconstruct_labels(pred)
    acc_arr.append(Metrics.accuracy(pred, orig))
    DSC_arr.append(Metrics.DSC(pred, orig, 2))
    DSC_core_arr.append(Metrics.DSC_core_tumor(pred, orig))
    PPV_arr.append(Metrics.PPV(pred, orig))
    print 'acc: {}'.format(acc_arr[i])
    print 'DSC: {}'.format(DSC_arr[i])
    print 'DSC_core: {}'.format(DSC_core_arr[i])
    print 'PPV : {}'.format(PPV_arr[i])
    sys.stdout.flush()
    final_pth = os.path.dirname(tst[4]) +  '/' + os.path.splitext(os.path.basename(tst[0]))[0] + '_predicted_70.mha'  
    pred_pth.append(final_pth)
    pred_arr.append(pred)
    t1c_pth.append([flp for flp in glob(os.path.dirname(tst[2]) + '/*.mha') if 'n4' not in flp])
               '''