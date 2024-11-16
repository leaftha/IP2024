# IP2024

## homework1 

```
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('cat.jpg')
original_img = img.copy()  
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
font = cv2.FONT_HERSHEY_SIMPLEX
value = 0
def update_value(x):
    global value
    value = x
# mouse callback function
def draw_circle(event,x,y,flags,param):

    global ix, iy, drawing, mode, img
    if event == cv2.EVENT_LBUTTONDOWN:
        img = original_img.copy()
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.putText(img,'Mouse Position (' + str(ix) +"," +str(iy) +")",(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
 
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            x_min, x_max = min(ix, x), max(ix, x)
            y_min, y_max = min(iy, y), max(iy, y)
        
            img[y_min:y_max, x_min:x_max, 0] = original_img[y_min:y_max, x_min:x_max, 0]  
            img[y_min:y_max, x_min:x_max, 1] = original_img[y_min:y_max, x_min:x_max, 1]  
            img[y_min:y_max, x_min:x_max, 2] = value  
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
img = cv2.imread('cat.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


cv2.createTrackbar('value', 'image', 0, 255, update_value)  # R 채널 범위 0~255

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv2.destroyAllWindows()

```
![SnapSave io-homework1](https://github.com/user-attachments/assets/41f2e220-9011-41a5-82df-85f838fc1e24)


## homework2

```
import cv2 
import numpy as np 
cap = cv2.VideoCapture(0) 
while(1): # Take each frame 
    _, frame = cap.read() # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # define range of blue color in HSV 
    lower_blue = np.array([110,50,50]) 
    upper_blue = np.array([130,255,255]) # Threshold the HSV image to get only blue colors 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
    cv2.imshow('frame',frame) 
    cv2.imshow('mask',mask) 
    cv2.imshow('res',res) 
    k = cv2.waitKey(5) & 0xFF 
    if k == 27: 
        break 
cv2.destroyAllWindows()
```
![Desktop 2024 10 13 - 14 02 07 01 (1)](https://github.com/user-attachments/assets/bc984da5-69ea-444e-9c99-97e5267f8ad7)


## homework3

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dsu2.jpg')
rows,cols,ch = img.shape
pts1 = np.float32([[180,241],[827,218],[813,511],[181,501]])
pts2 = np.float32([[0,0],[600,0],[600,400],[0,400]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(600,400))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```
![123](https://github.com/user-attachments/assets/f3d04d5c-f993-4c84-b9eb-6dfeb34f382e)


## homework4

![homewor4-1](https://github.com/user-attachments/assets/af53849b-39f0-48b5-a507-903468dc31e4)
![homewor4-2](https://github.com/user-attachments/assets/7bee93c6-295e-43cb-b216-5ad951c05df5)
![homewor4-3](https://github.com/user-attachments/assets/91f86f49-1c98-4c1b-8cb4-66d79d9c92e4)


