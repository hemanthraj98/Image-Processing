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
![op1](https://user-images.githubusercontent.com/72369402/105165079-e5b56300-5b3b-11eb-9756-e4dcefaae1dd.png)
![op2](https://user-images.githubusercontent.com/72369402/105165107-ec43da80-5b3b-11eb-8fa2-959c0dc197ef.png)


import cv2
import numpy as np
img=cv2.imread("flower1.jpg")
(rows,cols) = img.shape[:2]
M=cv2.getRotationMatrix2D((cols / 2, rows / 2),135,1)
res=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("result.jpg",res)
cv2.waitKey(0)
![op3](https://user-images.githubusercontent.com/72369402/105165941-f61a0d80-5b3c-11eb-9089-980c40bb5411.PNG)
![op4](https://user-images.githubusercontent.com/72369402/105166215-409b8a00-5b3d-11eb-8f2e-0786ac1c5f64.PNG)

import cv2
img = cv2.imread('flower2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![op](https://user-images.githubusercontent.com/72369402/105164279-f6190e00-5b3a-11eb-8b2f-4c34fc15cd50.png)
