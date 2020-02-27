#!/usr/bin/env python
# coding: utf-8

# ## Vectorly task Solution

# In[1]:


## Importing neccessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


## Visualizing original image
image = cv2.imread('simpsons_frame0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[3]:


## Making a copy of original image 
image_copy = np.copy(image)


# In[5]:


## Making masking image
lower_bound = np.array([0,0,0])
upper_bound = np.array([255,255,20])
masked_image = cv2.inRange(image_copy, lower_bound, upper_bound)
plt.imshow(masked_image, cmap = "Greys_r")


# In[7]:


## Applying threshold function
res, threshold_image = cv2.threshold(masked_image, 120, 255, cv2.THRESH_BINARY_INV)
plt.imshow(threshold_image, cmap = "Greys_r")


# In[8]:


## Applying median filter for getting clear image
median_image = cv2.medianBlur(threshold_image,15)
plt.imshow(median_image, cmap = "Greys_r")


# In[10]:


## Finally visualing origianl image, threshold image and median image(resultant image)
f, (ax1, ax2, ax3) = plt.subplots(1,3 ,figsize = (20, 10))

ax1.set_title('Orginal Image')
ax1.imshow(image)

ax2.set_title('Resultant Image before applying median filter')
ax2.imshow(threshold_image, cmap = "Greys_r")

ax3.set_title('Resultant Image after applying median filter')
ax3.imshow(median_image, cmap = "Greys_r")


# In[11]:


## Saving the resultant image
cv2.imwrite('simpons_text.png', median_image)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




