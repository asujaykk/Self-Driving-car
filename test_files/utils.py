#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:27:34 2022

@author: akhil_kk
"""

import pandas as pd
import cv2

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers

class dataset(keras.utils.Sequence) :

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
        y[i]=(batch_data.iloc[i,3])       
      return x,y
   
 
#img_shape=(160,320)
#batchsize=32
   
#train_ds=dataset(batchsize,'driving_log_cleared.csv',13610,img_shape)

      
