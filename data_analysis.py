import cv2
import numpy as np
import pandas as pd
data=pd.read_csv("driving_log_cleared.csv",header=0,names=['imgc','imgl','imgr','angle','par1','par2','spd'])
print(data.head())
print(data.info())
print(data.head())
print(data.describe())
print(data.angle ==0)        
      
      
      
      
