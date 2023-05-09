#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2
import numpy as np 


# In[3]:


# open a connection to the camera

cap = cv2.VideoCapture(0)


# In[4]:


# check if camera opened successfully
if not cap.isOpened():
  print("Error: Could not open camera.")
  exit()


while True:
  #Capture frame from camera
  ret, frame = cap.read()

  #Convert the frame to gray scale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #apply Gaussian blur
  blur = cv2.GaussianBlur(frame,(5,5),0)

  #apply Median blur
  median = cv2.medianBlur(frame, 5)

  #Apply Bilateral filter
  bilateral = cv2.bilateralFilter(frame, 15, 75,75)
    
  #RBG image
  RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
  # Box_Filtering
  box_filtering = cv2.boxFilter(frame, -1, (5,5), normalize=True, borderType=cv2.BORDER_DEFAULT)

    # 2D Filtering
    # Creating the kernel(2d convolution matrix)
  kernel1 = np.ones((5, 5), np.float32)/30
  d_filtering = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel1) 
    
   # threshold filtering
  retval,threshold = cv2.threshold(frame, 62, 255, cv2.THRESH_BINARY_INV)
    # Dilation
  kernel = np.ones((5,5), np.uint8)
  dilation = cv2.dilate(frame,kernel, iterations = 1)
    #erosion
  erosion = cv2.erode(frame, kernel, iterations=1)

  #Display the original frame and filtered frames
  cv2.imshow("Orginal",frame)
  cv2.imshow("Grayscale", gray)
  cv2.imshow("Gaussian Blur", blur)
  cv2.imshow("Median Blur", median)
  cv2.imshow("Bilateral Filter", bilateral)
  cv2.imshow("RGB Filter", RGB_img)
  cv2.imshow("Box Filter", box_filtering)
  cv2.imshow("2D Filtering", d_filtering)
  cv2.imshow("Threshold", threshold) 
  cv2.imshow("Dilation", dilation)
  cv2.imshow("Erosion", erosion)

  #Exit the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


# In[5]:


#Release the camera and close all the windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




