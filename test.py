#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:30:56 2022

@author: akhil_kk
"""

import pandas as pd
import cv2
data=pd.read_csv("test_data.csv")

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from utils import dataset

img_shape=(160,320)
batchsize=1
   
data_path='/home/akhil_kk/WORKING_PROJECT/car_sim_udacity/dataset/' 
train_ds=dataset(batchsize,data_path+'track1/driving_log_cleared.csv',13610,img_shape)
test_ds=dataset(batchsize,data_path+'track2/driving_log.csv',3729,img_shape)
x,y=train_ds.__getitem__(4)
print(train_ds.__len__())
print(y)
cv2.imshow("fra",x[0])
cv2.waitKey(2000)
 


premodel='models/save_at_14.h5'
model_1=keras.models.load_model(premodel)

for i in range(13610):
   x,y=train_ds.__getitem__(i)
   output=model_1.predict(x)
   image_array=x[0]
   cv2.imshow("frame",np.dstack((image_array[:,:,2],image_array[:,:,1],image_array[:,:,0])))
   print("input: "+str(y[0])+" output: "+str(output/25.00))
   cv2.waitKey(1)