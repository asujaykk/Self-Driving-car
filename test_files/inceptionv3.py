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


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024,acttivation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
x = Dense(64,acttivation='relu')(x)
predictions = Dense(1,activation='tanh')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# train the model on the new data for a few epochs
#model.fit(...)  
 
img_shape=(160,320)
batchsize=8
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
       "huber_loss",
       "log_cosh",
        ]

lrs=[0.001,0.0001,0.00001,0.01]
#model.build(img_shape)
model.summary()


epochs = 50

x=np.arange(0,epochs,1)


for lr in lrs:
    for loss in losses:
        callbacks = [
            keras.callbacks.ModelCheckpoint("./models/"+str(lr)+"/"+loss+"/save_at_{epoch}.h5"),
            keras.callbacks.EarlyStopping(monitor='loss',patience=5,mode='min'),
            keras.callbacks.TensorBoard(log_dir="./logs/"+str(lr)+"/"+loss)
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
        

      
