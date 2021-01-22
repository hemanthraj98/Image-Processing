# Image-Processing
**1. Develop a program to display grayscale image using read and write operation.
**
**Description:**
    Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white. 
imread() : is used for reading an image.
imwrite(): is used to write an image in memory to disk.
imshow() :to display an image.
waitKey(): The function waits for specified milliseconds for any keyboard event.
destroyAllWindows():function to close all the windows.
cv2. cvtColor() method is used to convert an image from one color space to another
    syntax is cv2.cvtColor(Input_image,flag)
**Program:**

import cv2
import numpy as np
image=cv2.imread("flower.jpg")
cv2.imshow("Old",image)
cv2.imshow("Gray",grey)
grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray",grey)
cv2.imwrite("flower.jpg",grey)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**


![output](https://user-images.githubusercontent.com/72369402/105163284-ba317900-5b39-11eb-9103-313528df9fee.png)

**2. Develop a program to perform linear transformations on an image: Scaling and Rotation**
**Description:**
  **A)Scaling:** Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machinelearning applications. 
It helps in reducing the number of pixels from an image cv2.
resize() method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
imshow() function in pyplot module of matplotlib library is used to display data as an image
**Program**
import cv2
import numpy as np
img=cv2.imread('flower1.jpg')
(height,width)=img.shape[:2]
res=cv2.resize(img,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('result.jpg',res)
cv2.imshow('image',img)
cv2.imshow('result',res)
cv2.waitKey(0)

**output**

![op1](https://user-images.githubusercontent.com/72369402/105165079-e5b56300-5b3b-11eb-9756-e4dcefaae1dd.png)
![op2](https://user-images.githubusercontent.com/72369402/105165107-ec43da80-5b3b-11eb-8fa2-959c0dc197ef.png)


**B)Rotation:** 
         **Description:** 
                Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal,flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing. 
cv2.getRotationMatrix2D Perform the counter clockwise rotation warpAffine() function is the size of the output image, which should be in the form of (width, height).
width = number of columns, and height = number of rows.
**Program**
import cv2
import numpy as np
img=cv2.imread("flower1.jpg")
(rows,cols) = img.shape[:2]
M=cv2.getRotationMatrix2D((cols / 2, rows / 2),135,1)
res=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("result.jpg",res)
cv2.waitKey(0)

**output**

![op](https://user-images.githubusercontent.com/72369402/105164279-f6190e00-5b3a-11eb-8b2f-4c34fc15cd50.png)


**4. Develop a program to convert the color image to gray scale and binary image.**
**Description:** 
        Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. 
It varies between complete black and
complete white. A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, 
usually black and white. 

**Program**
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

**output**

![op3](https://user-images.githubusercontent.com/72369402/105165941-f61a0d80-5b3c-11eb-9089-980c40bb5411.PNG)
![op4](https://user-images.githubusercontent.com/72369402/105166215-409b8a00-5b3d-11eb-8f2e-0786ac1c5f64.PNG)

**5. Develop a program to convert the given color image to different color spaces.**
**Description:**
        Color spaces are a way to represent the color channels present in the image that gives the image that particular hue 
BGR color space: OpenCV’s default color space is RGB. 
HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye.
Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. 
LAB color space :
       L – Represents Lightness.
       A – Color component ranging from Green to Magenta.
       B – Color component ranging from Blue to Yellow. 
The HSL color space, also called HLS or HSI, stands for:
Hue : the color type Ranges from 0 to 360° in most applications 
Saturation : variation of the color depending on the lightness. 
Lightness :(also Luminance or Luminosity or Intensity). Ranges from 0 to 100% (from black to white).
YUV:Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives intensity information very differently from color information.
cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white),else it is assigned another value (may be black). destroyAllWindows() simply destroys all the windows we created. To destroy any specific window, use the function 
cv2.destroyWindow() where you pass the exact window name.
**Program**
import cv2
img = cv2.imread('flower2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow('GRAY image',gray)
cv2.waitKey(0)
cv2.imshow('HSV image',hsv)
cv2.waitKey(0)
cv2.imshow('LAB image',lab)
cv2.waitKey(0)
cv2.imshow('HLS image',hls)
cv2.waitKey(0)
cv2.imshow('YUV image',yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.destroyAllWindows()

**output**

![op5](https://user-images.githubusercontent.com/72369402/105168707-8148d280-5b40-11eb-8e77-20a1c5684517.PNG)
![hsv](https://user-images.githubusercontent.com/72369402/105168804-a3daeb80-5b40-11eb-807b-54835212a5ff.PNG)
![lab](https://user-images.githubusercontent.com/72369402/105168830-afc6ad80-5b40-11eb-8adc-95372ccb2ecb.PNG)
![hls](https://user-images.githubusercontent.com/72369402/105168873-bd7c3300-5b40-11eb-8583-8546162017f9.PNG)
![yuv](https://user-images.githubusercontent.com/72369402/105168954-d97fd480-5b40-11eb-9fef-9a73237ff0ec.PNG)


