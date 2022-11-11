
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:32:23 2022

@author: akhil_kk

This file contain a method to create a keras model for self driving car project
   Keras model contain
   1. Three 5x5 kernal stride 2 CNN layers
   2. Two 3x3 kernal stride 1 cnn layers
   3. And a fully connected layers haveing one output.



The method takes one agrument 'inshape'
   This should mention (height , width) of the input image.
   The output of the model will be a single float value ranging from (-1 to 1) : this will be the steering angle to be provided to the simulator
   The method return a keras model

"""


#import required modules
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

 

#method to create the model
def make_model(inshape):
   """
    This model create a CNN model with a single regression output 
    Arguments:  image shape:  (height,width)
    returns: A keras model
   """
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
   outputs= layers.Dense(1,activation="tanh")(x)     # tanh activation used since output range from -1 to 1
   
   return keras.Model(inputs,outputs)

        

      
