
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:32:23 2022

@author: akhil_kk
"""

import pandas as pd
#import cv2
data=pd.read_csv("test_data.csv")

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from utils import dataset
 

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Activation


def make_model2(inshape):
   inputs = keras.Input(shape=inshape+(3,))
   x = layers.Rescaling(1.0 / 255.0)(inputs)
   x = layers.Conv2D(24, 5, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(36, 5, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(48, 5, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(78, 3, strides=1, padding="same")(x)
   x = layers.Activation("relu")(x)  
   x = layers.Flatten()(x)
   x= layers.Dense(100,activation="relu")(x)
   x= layers.Dense(50,activation="relu")(x)  
   x= layers.Dense(10,activation="relu")(x) 
   outputs= layers.Dense(1,activation="tanh")(x) 
   return keras.Model(inputs,outputs)

img_shape=(160,320)
batchsize=8
model=make_model2( inshape=(img_shape[0],img_shape[1]))
#model.build(img_shape)
model.summary()
 

data_path='/home/akhil_kk/WORKING_PROJECT/car_sim_udacity/dataset/' 
train_ds=dataset(batchsize,data_path+'track1/driving_log_cleared.csv',13610,img_shape)
test_ds=dataset(batchsize,data_path+'track2/driving_log.csv',3729,img_shape)
x,y=train_ds.__getitem__(4)
print(train_ds.__len__())
print(y)
#cv2.imshow("fra",len(item))
#cv2.waitKey(5000)
 



losses=[
       "mean_squared_error",
       "mean_absolute_error",
       "mean_absolute_percentage_error",
       "mean_squared_logarithmic_error",
       "cosine_similarity",
       "log_cosh",
        ]

lrs=[0.001,0.0001,0.00001,0.01]
#model.build(img_shape)
model.summary()


epochs = 50

x=np.arange(0,epochs,1)


for lr in lrs:
    if lr==0.001:
        continue
    
    for loss in losses:
        print("learning rate: "+str(lr))
        print("loss : "+loss)
        callbacks = [
            keras.callbacks.ModelCheckpoint("./models_cnn/"+str(lr)+"/"+loss+"/save_at_{epoch}.h5"),
            keras.callbacks.EarlyStopping(monitor='loss',patience=5,mode='min'),
            keras.callbacks.TensorBoard(log_dir="./logs_cnn/"+str(lr)+"/"+loss)
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=loss,
            metrics=loss,
        )
                
        result=model.fit(
            train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks,
        )
        
        # t_loss=result.history['loss']
        # t_metr_loss=result.history[loss]

        # v_loss=result.history['val_loss']
        # v_metr_loss=result.history['val_'+loss]
        
        # plt.plot(x,t_loss,'or')
        # plt.plot(x,t_metr_loss,'og')
        # plt.plot(x,v_loss,'ob')
        # plt.plot(x,v_metr_loss,'oy')
        # plt.savefig('result_plots/'+str(lr)+"/"+loss+"/result.png")
        # plt.clf()
        

      
