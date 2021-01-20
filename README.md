# Image-Processing
import cv2
import numpy as np
image=cv2.imread("flower1.jpg")
image=cv2.resize(image,(0,0),None,.95,.95)
grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
grey_3_channel=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
numpy_horizontal=np.hstack((image,grey_3_channel))
numpy_horizontal_concat=np.concatenate((image,grey_3_channel),axis=1)
cv2.imshow("flower",numpy_horizontal_concat)
cv2.waitKey()


![output](https://user-images.githubusercontent.com/72369402/105163284-ba317900-5b39-11eb-9103-313528df9fee.png)

import cv2
import numpy as np
img=cv2.imread('flower1.jpg')
(height,width)=img.shape[:2]
res=cv2.resize(img,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('result.jpg',res)
cv2.imshow('image',img)
cv2.imshow('result',res)
cv2.waitKey(0)


import cv2
import numpy as np
img=cv2.imread("flower1.jpg")
(rows,cols) = img.shape[:2]
M=cv2.getRotationMatrix2D((cols / 2, rows / 2),135,1)
res=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("result.jpg",res)
cv2.waitKey(0)

![op](https://user-images.githubusercontent.com/72369402/105164279-f6190e00-5b3a-11eb-8b2f-4c34fc15cd50.png)
