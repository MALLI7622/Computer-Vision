
import cv2
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np



c = cv2.imread('c.jpg')
gray_c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)


plt.figure(figsize = (20,20))
plt.imshow(c)



plt.figure(figsize = (20,30))
plt.imshow(gray_c,cmap = 'gray')


cv2.imshow("Figure",gray_c)
cv2.waitKey()
cv2.destroyAllWindows()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')






face_detect = face_cascade.detectMultiScale(gray_c, scaleFactor = 1.1,minNeighbors = 5)



for ( x,y,w,h ) in face_detect:
    cv2.rectangle(c,(x,y),(x+w, y+h),(0,0,255),2)



plt.figure(figsize=(20,20))
plt.imshow(cv2.cvtColor(c,cv2.COLOR_BGR2RGB))
