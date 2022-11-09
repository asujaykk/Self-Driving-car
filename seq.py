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
batchsize=32
   
train_ds=data(batchsize,'driving_log_cleared.csv',13610,img_shape)
x,y=train_ds.__getitem__(4)
print(train_ds.__len__())
print(y)
#cv2.imshow("fra",len(item))
#cv2.waitKey(5000)
 
def make_model(inshape):
   inputs = keras.Input(shape=inshape+(3,))
   x = layers.Rescaling(1.0 / 255.0)(inputs)
   x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   for no_of_ch in (64,128,256):
       x=layers.Conv2D(no_of_ch,3,strides=2,padding="same")(x)
       x=layers.Activation("relu")(x)

   x = layers.Flatten()(x)
   x= layers.Dense(64,activation="tanh")(x)
   outputs= layers.Dense(1,activation="tanh")(x)    
   return keras.Model(inputs,outputs)

def make_model1(inshape):
   inputs = keras.Input(shape=inshape+(3,))
   x = layers.Rescaling(1.0 / 255.0)(inputs)
   x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(16, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Flatten()(x)
   x= layers.Dense(64,activation="tanh")(x)
   x= layers.BatchNormalization()(x)
   x= layers.Dense(1,activation="tanh")(x)    
   outputs= layers.BatchNormalization()(x)
   return keras.Model(inputs,outputs)

def make_model3(inshape):
   inputs = keras.Input(shape=inshape+(3,))
   x = layers.Rescaling(1.0 / 255.0)(inputs)
   x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(16, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
   x = layers.Activation("relu")(x)
   x = layers.Flatten()(x)
   x= layers.Dense(64,activation="tanh")(x)
   x= layers.BatchNormalization()(x)
   x= layers.Dense(1,activation="tanh")(x)    
   outputs= layers.BatchNormalization()(x)
   return keras.Model(inputs,outputs)


model=make_model1( inshape=(img_shape[0],img_shape[1]))
#model.build(img_shape)
model.summary()


epochs = 50
callbacks = [
    keras.callbacks.ModelCheckpoint("models/save_at_{epoch}.h5"),
    #keras.callbacks.EarlyStopping(monitor='accuracy',patience=3)
]
model.compile(
    optimizer=keras.optimizers.SGD(1e-3),
    loss="mean_absolute_error",
    metrics=["accuracy"],
)

ut=model.predict(x)
print(ut)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks,
)

      
