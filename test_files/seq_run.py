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
        y[i]=(batch_data.iloc[i,3])       
      return x,y
   
 
img_shape=(160,320)
batchsize=1
   
train_ds=data(batchsize,'driving_log_cleared.csv',13610,img_shape)
x,y=train_ds.__getitem__(4)
print(train_ds.__len__())
print(y)
cv2.imshow("fra",x[0])
cv2.waitKey(8000)
 


premodel='models/save_at_37.h5'
model_1=keras.models.load_model(premodel)

output=model_1.predict(train_ds.__getitem__(200)[0])
print(output)

      
