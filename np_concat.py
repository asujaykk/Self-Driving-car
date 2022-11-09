import cv2
import numpy as np

img1=cv2.imread('img/face1.jpeg')
img2=cv2.imread('img/face2.jpeg')
img3=cv2.imread('img/face3.jpeg')


img1=cv2.resize(img1,(100,100),interpolation=cv2.INTER_NEAREST)
img2=cv2.resize(img2,(100,100),interpolation=cv2.INTER_NEAREST)
img3=cv2.resize(img3,(100,100),interpolation=cv2.INTER_NEAREST)



imgf=np.concatenate((img1,img2,img3),axis=2)
#imgf=np.hstack((img1,img2,img3))


cv2.imshow('final',imgf)
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img3",img3)

cv2.waitKey(8000)
        
      
      
      
      
