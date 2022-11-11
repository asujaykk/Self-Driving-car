#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:27:34 2022

@author: akhil_kk

"""

import pandas as pd
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class dataset(keras.utils.Sequence) :
    """
        This module is a custom keras sequence generator (dataset)
    
    the datset is the class created from keras.utils.Sequence class
    
    The constructor of this class expect the following parameters:
    1. The CSV file path, which represent the driving log dataset
    2. batch size : default value is 32
    3. count: for selecting range of elements to be selected from the csv file
         * 0-count elements will be considered by the sequence generator (count should be less than total raws in csv file)
         * if count is not provided then whole elements will be considered by the generator
     
    4. imgsize: the size in which the image to be loaded (height,width)
          * if not provided then the img size will be derived from the image itself
    
    5. Output sequence will be
         x= centre camera image (numpy array)
         y= steering angle (float value)
    """
    
    def __init__(self,csvfile,batchsize=32,count=None,imgsize=None):
       
      self.batchsize=batchsize
      self.csvfile=csvfile
      self.data=pd.read_csv(self.csvfile)
      
      #print(self.data.shape)
      if count==None:
          self.count=self.data.shape[0]   # number of raws in the csv file
      else:
          self.count=count
      
      if imgsize==None:
        size=load_img(self.data.iloc[0,0]).size  #loaded PIL image size format is (width,height)
        self.img_size= (size[1],size[0])  # converting shape to (height,width ) format
      else:
          self.img_size=imgsize
    
    
    
    def __len__(self,):
        #batch count is tha data length
      return  (self.count // self.batchsize)
   
    
   
    def get_img_shape(self):    # return image shape in case user want original image shape
       return self.img_size
   
    
    def __getitem__(self,idx):
      i=idx*self.batchsize    # find the index of data in the csv file based on the batch size and requested index
      batch_data=self.data.iloc[i:i+self.batchsize]    # extract the batch size data 
      x=np.zeros((self.batchsize,)+(self.img_size)+(3,),dtype='uint8')  #numpy array of batch data shape 
      y=np.zeros((self.batchsize,)+(1,))
      for i in range(self.batchsize):    
        imgc=load_img(batch_data.iloc[i,0],target_size=self.img_size)  # load centre cam img
        #imgl=load_img(batch_data.iloc[i,1],target_size=self.img_size) # load left cam img
        #imgr=load_img(batch_data.iloc[i,2],target_size=self.img_size) # load right cam img
        #x[i]=np.hstack((imgl,imgc,imgr))    # stack image in horizontally
        x[i]=imgc                           # only put centre cam image in x array
        y[i]=(batch_data.iloc[i,3])         # only put steering angle as y value
      return x,y
   
 
#data_path='/home/akhil_kk/WORKING_PROJECT/car_sim_udacity/dataset/' 
#train_ds=dataset(data_path+'track1/driving_log_cleared.csv')

#print(train_ds.get_img_shape()) 
#print(train_ds.__len__()) 
 
#img_shape=(160,320)
#batchsize=32
   
#train_ds=dataset(batchsize,'driving_log_cleared.csv',13610,img_shape)

      
