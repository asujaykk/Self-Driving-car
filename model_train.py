
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:32:23 2022

@author: akhil_kk
"""
import os
import pandas as pd
from tensorflow import keras
from utils import dataset
from model import make_model
import argparse


parser = argparse.ArgumentParser(description = 'Train the model for self driving car')
parser.add_argument('--train_csv_file',metavar='string',type=str,required = True,help='Path to train driving log csv file')
parser.add_argument('--test_csv_file',metavar='string',type=str,required = True,help='Path to test driving log csv file')
parser.add_argument('--batch_size',metavar='int',type=int,default=32,help='batchsize for training, default: 32')
parser.add_argument('--epochs',metavar='int',type=int,default=50,help='epochs for training, default: 50')


args = parser.parse_args()


batchsize= args.batch_size
epochs = args.epochs

# path to test and training csv file
train_csv_file_path=args.train_csv_file
test_csv_file_path=args.test_csv_file


# Create training and testing data set 
train_ds=dataset(train_csv_file_path,batchsize)
if test_csv_file_path is not None:
   test_ds=dataset(test_csv_file_path,batchsize)
else:
    test_ds=None 


# get original image shape
img_shape= train_ds.get_img_shape()  

# make model

model=make_model( inshape=img_shape)  
model.summary()   #print model summary


# loss function
loss="mean_squared_error"        


# train the model for all learning rate 
# and save the model with minimum loss
lrs=[0.001,0.0001,0.00001,0.01] 



#train the model for all learning rates
for lr in lrs:
        print("learning rate: "+str(lr))
        print("loss : "+loss)
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join('models',str(lr),str(loss),'save_at_{epoch}.h5'),save_best_only= True),
            keras.callbacks.EarlyStopping(monitor='loss',patience=5,mode='min'),
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=loss,
            metrics=loss,
        )
                
        result=model.fit(
            train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks,
        )
        
        

      
