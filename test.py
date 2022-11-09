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

class data(keras.utils.Sequence) :

   def __init__(self,batchsize,csvfile,count,imgsize):
      self.batchsize=batchsize
      self.csvfile=csvfile
      self.count=count
      self.data=pd.read_csv(self.csvfile)
      self.img_size=imgsize
          
   def __len__(self,):
      return  (self.count // self.batchsize)
   
   
   def __getitem__(self,idx):
      i=idx*self.batchsize
      batch_data=self.data.iloc[i:i+self.batchsize]
      x=np.zeros((self.batchsize,)+(self.img_size)+(3,),dtype='uint8')
      y=np.zeros((self.batchsize,)+(1,))
      for i in range(self.batchsize):    
        imgc=load_img(batch_data.iloc[i,0],target_size=self.img_size)
        #imgl=load_img(batch_data.iloc[i,1],target_size=self.img_size)
        #imgr=load_img(batch_data.iloc[i,2],target_size=self.img_size)
        #x[i]=np.hstack((imgl,imgc,imgr))
        x[i]=imgc
        y[i]=(batch_data.iloc[i,3]*25.00)       
      return x,y
   
 
img_shape=(160,320)
batchsize=1
   
train_ds=data(batchsize,'driving_log_cleared.csv',13610,img_shape)
x,y=train_ds.__getitem__(4)
print(train_ds.__len__())
print(y)
cv2.imshow("fra",x[0])
cv2.waitKey(2000)
 


premodel='models/save_at_4.h5'
model_1=keras.models.load_model(premodel)

for i in range(13610):
   x,y=train_ds.__getitem__(i)
   output=model_1.predict(x)
   image_array=x[0]
   cv2.imshow("frame",np.dstack((image_array[:,:,2],image_array[:,:,1],image_array[:,:,0])))
   print("input: "+str(y[0])+" output: "+str(output/25.00))
   cv2.waitKey(1)